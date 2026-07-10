{
  description = "omnirun - run jobs anywhere, cheaply";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    git-hooks.url = "github:cachix/git-hooks.nix";
  };

  outputs = { self, nixpkgs, flake-utils, git-hooks }:
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
      });
}
