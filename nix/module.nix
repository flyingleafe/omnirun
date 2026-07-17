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

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.omnirun;
      defaultText = lib.literalExpression "pkgs.omnirun";
      description = "The omnirun package to run (from this flake's overlay).";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "omnirun";
      description = "System user the daemon runs as.";
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
  };

  config = lib.mkIf cfg.enable {
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.user;
      home = cfg.stateDir;
      createHome = true;
      description = "omnirun scheduler daemon";
    };
    users.groups.${cfg.user} = { };

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
      };
      serviceConfig = {
        ExecStart = "${cfg.package}/bin/omnirun serve";
        Restart = "on-failure";
        RestartSec = "5";
        User = cfg.user;
        Group = cfg.user;
        StateDirectory = "omnirun";
        EnvironmentFile = lib.mkIf (cfg.environmentFile != null) cfg.environmentFile;
      };
    };
  };
}
