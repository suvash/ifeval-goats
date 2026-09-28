{
  description = "Python + uv dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.nodejs
            pkgs.uv
          ];

          shellHook = ''
            # Keep uv-installed tools project-local instead of ~/.local/{bin,share}
            export UV_TOOL_DIR="$PWD/.uv/tools"
            export UV_TOOL_BIN_DIR="$PWD/.uv/bin"
            export PATH="$UV_TOOL_BIN_DIR:$PATH"

            # Create a local venv on first run (uv manages the Python interpreter itself)
            if [ ! -d .venv ]; then
              uv venv .venv
            fi
            source .venv/bin/activate

            # Ensure the 'prime' CLI tool is installed
            if ! uv tool list 2>/dev/null | grep -q '^prime'; then
              uv tool install prime
            fi

            echo "Python: $(python --version)"
            echo "uv $(uv --version) ready."
          '';
        };
      });
}
