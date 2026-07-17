{
  description = "omnirun - run jobs anywhere, cheaply";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    git-hooks.url = "github:cachix/git-hooks.nix";
  };

  outputs = { self, nixpkgs, flake-utils, git-hooks }:
    let
      # The omnirun package. Core deps + the daemon/postgres/kaggle extras are all
      # in nixpkgs; the colab CLI is a runtime *binary* (not a Python dep), so the
      # package builds without it — add it via `services.omnirun.extraPackages`.
      # The `colab` CLI the Colab backend shells out to (packaged here — not in
      # nixpkgs). Wrapped onto omnirun's PATH so the daemon has Colab support.
      mkColabCli = pkgs: pkgs.callPackage ./nix/colab-cli.nix { };

      # kaggle 2.2.x (nixpkgs ships an ancient 1.7.4.5 lacking OAuth support),
      # built for omnirun's python since omnirun imports it.
      mkKaggle = pkgs: pkgs.callPackage ./nix/kaggle.nix {
        python3Packages = pkgs.python312Packages;
      };

      mkOmnirun = pkgs: pkgs.python312Packages.buildPythonApplication {
        pname = "omnirun";
        # Single source of truth: read the version from pyproject.toml so the
        # nix label never drifts from __version__ on a release bump.
        version = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project.version;
        pyproject = true;
        src = self;
        build-system = [ pkgs.python312Packages.hatchling ];
        dependencies = (with pkgs.python312Packages; [
          typer
          rich
          httpx
          pydantic
          sqlalchemy
          bottle
          psycopg
        ]) ++ [ (mkKaggle pkgs) ];
        nativeBuildInputs = [ pkgs.makeWrapper ];
        # Tests are live-gated + run in CI; skip them in the build sandbox.
        doCheck = false;
        # Runtime helper binaries the daemon/CLI shells out to (the Colab backend
        # invokes `colab`; ssh/slurm/deploy-key work needs git/gh/ssh/rsync; the
        # worker env build uses uv).
        postFixup = ''
          wrapProgram $out/bin/omnirun \
            --prefix PATH : ${pkgs.lib.makeBinPath [
              pkgs.git
              pkgs.gh
              pkgs.openssh
              pkgs.rsync
              pkgs.uv
              (mkColabCli pkgs)
            ]}
        '';
        meta = with pkgs.lib; {
          description = "Run jobs from your repo anywhere: Slurm/SSH/Kaggle/Colab/marketplace GPUs";
          homepage = "https://github.com/flyingleafe/omnirun";
          license = licenses.mit;
          mainProgram = "omnirun";
        };
      };
    in
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;

        # Hooks that need no project dependencies, so they also work inside
        # the hermetic `nix flake check` sandbox (no venv, no network).
        sandboxHooks = {
          ruff = {
            enable = true;
            package = pkgs.ruff;
          };
          ruff-format = {
            enable = true;
            package = pkgs.ruff;
          };
        };

        # `nix flake check` runs this: sandbox-safe hooks only.
        ci-check = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = sandboxHooks;
        };

        # Local `git commit` hook set: adds basedpyright, which needs
        # numpy/hypothesis from the uv venv activated by the devShell
        # hook below, so it can't run inside the flake-check sandbox.
        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = sandboxHooks // {
            pyright = {
              enable = true;
              package = pkgs.basedpyright;
              settings.binPath = "${pkgs.basedpyright}/bin/basedpyright";
            };
          };
        };
      in
      {
        packages.default = mkOmnirun pkgs;
        packages.omnirun = mkOmnirun pkgs;
        packages.google-colab-cli = mkColabCli pkgs;
        # The proved trace checker (formal/ Lean model, DEPLOY-V2.md §1):
        # validates job_events traces against the formal kernel model. Used
        # by the omnirun-validator service and the conformance test suites.
        packages.trace-check = pkgs.stdenv.mkDerivation {
          pname = "omnirun-trace-check";
          version = "0.1.0";
          src = ./formal;
          nativeBuildInputs = [ pkgs.lean4 ];
          buildPhase = ''
            export HOME=$TMPDIR
            lake build trace-check
          '';
          installPhase = ''
            install -Dm755 .lake/build/bin/trace-check $out/bin/trace-check
          '';
        };

        formatter =
          let
            inherit (pre-commit-check.config) package configFile;
            script = ''
              ${pkgs.lib.getExe package} run --all-files --config ${configFile}
            '';
          in
          pkgs.writeShellScriptBin "pre-commit-run" script;

        checks = {
          inherit ci-check;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = pre-commit-check.enabledPackages ++ (with pkgs; [
            python
            uv
            # C++ standard library for NumPy and other native dependencies
            stdenv.cc.cc.lib
            zlib
          ]);

          shellHook = ''
            ${pre-commit-check.shellHook}
            # Guard against tagging a release without bumping the version.
            # pre-commit owns only the pre-commit hook, so pre-push is ours.
            ln -sf "$PWD/scripts/pre-push" "$(git rev-parse --git-path hooks)/pre-push"
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:$LD_LIBRARY_PATH
            export UV_PYTHON=${python}/bin/python
            uv sync --quiet
            source .venv/bin/activate
          '';
        };
      })
    // {
      # System-agnostic outputs: an overlay that adds `omnirun` to a pkgs set,
      # and the NixOS module that runs the daemon (`services.omnirun`).
      overlays.default = final: _prev: { omnirun = mkOmnirun final; };
      nixosModules.default = import ./nix/module.nix;
    };
}
