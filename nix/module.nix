# NixOS module for the optional omnirun scheduler daemon (`omnirun serve`).
# Enable with `services.omnirun.enable = true` after adding this flake's
# `nixosModules.default`. The daemon owns the state store and all backend
# credentials; point clients at it with `[daemon].address` in their config.
{ config, lib, pkgs, ... }:
let
  cfg = config.services.omnirun;
in
{
  options.services.omnirun = {
    enable = lib.mkEnableOption "the omnirun scheduler daemon";

    validator = {
      enable = lib.mkEnableOption ''
        the replay validator (DEPLOY-V2 §1): tails job_events, re-validates
        both trace views through the proved trace-check binary each round,
        and files a deduplicated GitHub issue per formal-model violation
      '';
      traceCheck = lib.mkOption {
        type = lib.types.package;
        default = pkgs.omnirun-trace-check;
        defaultText = lib.literalExpression "pkgs.omnirun-trace-check";
        description = "The trace-check package (this flake's overlay provides it).";
      };
      intervalS = lib.mkOption {
        type = lib.types.int;
        default = 60;
        description = "Seconds between validation rounds.";
      };
      extraArgs = lib.mkOption {
        type = lib.types.listOf lib.types.str;
        default = [ ];
        description = "Extra arguments for `omnirun validate-replay` (e.g. --dry-run).";
      };
      ghRepo = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "owner/repo";
        description = ''
          GitHub repository (owner/repo) violations are filed against, exported
          as GH_REPO — required because the validator runs outside any git
          checkout and `gh` cannot infer a repo from its cwd.
        '';
      };
    };

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.omnirun;
      defaultText = lib.literalExpression "pkgs.omnirun";
      description = "The omnirun package to run (from this flake's overlay).";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "omnirun";
      description = "User the daemon runs as.";
    };

    createUser = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to create `user` as a dedicated system user. Set false to run as
        an EXISTING user (e.g. a login user that already holds the backend
        credentials in its home — kaggle/colab/gh/ssh config), so the daemon
        inherits them instead of re-provisioning secrets.
      '';
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = cfg.user;
      defaultText = lib.literalExpression "config.services.omnirun.user";
      description = "Group the daemon runs as (an existing user's primary group when createUser = false).";
    };

    stateDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/omnirun";
      description = ''
        OMNIRUN_STATE_DIR — where the daemon keeps job state, durable logs, and
        cached outputs (SQLite lives here unless the config points [state] at a
        Postgres URL). Created as a systemd StateDirectory.
      '';
    };

    configFile = lib.mkOption {
      type = lib.types.path;
      description = ''
        OMNIRUN_CONFIG — the daemon's TOML config: [daemon] bind host/port,
        [backends.*], [state] store URL, budgets. Backends' secrets belong in
        `environmentFile`, not here.
      '';
    };

    tmpDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/omnirun/tmp";
      description = ''
        TMPDIR for the daemon — where pull staging (`omnirun-pull-*`), snapshot
        bundles, and other tempfile work land. The default keeps them on real
        disk under the state dir instead of the host's /tmp, which on typical
        VPS hosts is a small tmpfs that large job outputs fill (ENOSPC on
        pull/submit staging). Created by tmpfiles; entries older than 3 days
        are cleaned.
      '';
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/run/secrets/omnirun.env";
      description = ''
        systemd EnvironmentFile with the daemon's secrets as KEY=value lines —
        backend API keys, and (for a Postgres store) PGPASSWORD or a ~/.pgpass
        reference. Typically a sops-nix-decrypted path. Never world-readable.
      '';
    };

    extraPackages = lib.mkOption {
      type = lib.types.listOf lib.types.package;
      default = [ ];
      example = lib.literalExpression "[ pkgs.google-colab-cli ]";
      description = ''
        Extra runtime binaries to put on the daemon's PATH — e.g. a provider CLI
        (`colab`) not shipped with omnirun. git/gh/openssh/rsync/uv are already
        included.
      '';
    };

    logLevel = lib.mkOption {
      type = lib.types.enum [ "debug" "info" "warning" "error" ];
      default = "info";
      example = "debug";
      description = ''
        Daemon log verbosity (journald). `debug` traces every backend/API/ssh
        action — each ssh command and its stderr, each provisioning poll's
        instance status — so a stuck placement is fully diagnosable.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    # Daemon TMPDIR on real disk (see tmpDir option); age out stale staging.
    systemd.tmpfiles.rules = [
      "d ${cfg.tmpDir} 0750 ${cfg.user} ${cfg.group} 3d"
    ];

    users.users = lib.mkIf cfg.createUser {
      ${cfg.user} = {
        isSystemUser = true;
        group = cfg.group;
        home = cfg.stateDir;
        createHome = true;
        description = "omnirun scheduler daemon";
      };
    };
    users.groups = lib.mkIf cfg.createUser { ${cfg.group} = { }; };

    systemd.services.omnirun = {
      description = "omnirun scheduler daemon";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      # The daemon shells out to these for client-side git work, deploy-key
      # provisioning, ssh/slurm transport, and worker env builds.
      path = [
        cfg.package
        pkgs.git
        pkgs.gh
        pkgs.openssh
        pkgs.rsync
        pkgs.uv
      ] ++ cfg.extraPackages;
      environment = {
        OMNIRUN_CONFIG = toString cfg.configFile;
        OMNIRUN_STATE_DIR = cfg.stateDir;
        OMNIRUN_LOG_LEVEL = cfg.logLevel;
        TMPDIR = cfg.tmpDir;
      };
      serviceConfig = {
        ExecStart = "${cfg.package}/bin/omnirun serve";
        Restart = "on-failure";
        RestartSec = "5";
        User = cfg.user;
        Group = cfg.group;
        StateDirectory = "omnirun";
        EnvironmentFile = lib.mkIf (cfg.environmentFile != null) cfg.environmentFile;
      };
    };

    # The replay validator: a separate unit so a validator crash never touches
    # the daemon. Same user (store access + `gh` auth from $HOME); needs git+gh
    # on PATH for issue filing.
    systemd.services.omnirun-validator = lib.mkIf cfg.validator.enable {
      description = "omnirun replay validator (formal-model conformance)";
      wantedBy = [ "multi-user.target" ];
      after = [ "omnirun.service" ];
      path = [ cfg.package pkgs.git pkgs.gh ];
      environment = {
        OMNIRUN_CONFIG = toString cfg.configFile;
        OMNIRUN_STATE_DIR = cfg.stateDir;
        OMNIRUN_LOG_LEVEL = cfg.logLevel;
        OMNIRUN_TRACE_CHECK = "${cfg.validator.traceCheck}/bin/trace-check";
      } // lib.optionalAttrs (cfg.validator.ghRepo != null) {
        GH_REPO = cfg.validator.ghRepo;
      };
      serviceConfig = {
        ExecStart = lib.concatStringsSep " " ([
          "${cfg.package}/bin/omnirun"
          "validate-replay"
          "--interval"
          (toString cfg.validator.intervalS)
        ] ++ cfg.validator.extraArgs);
        Restart = "on-failure";
        RestartSec = "30";
        User = cfg.user;
        Group = cfg.group;
        EnvironmentFile = lib.mkIf (cfg.environmentFile != null) cfg.environmentFile;
      };
    };
  };
}
