{
  description = "MaLO - Multi-agent framework for offensive security";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python311;
        
        pythonEnv = python.withPackages (ps: with ps; [
          # Core dependencies from environment.yml
          flask
          openai
          rich
          pytest
          pyyaml
          docker
          click
          
          # Build tools
          pip
          virtualenv
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
            git
          ];

          shellHook = ''
            echo "Setting up MaLO development environment..."
            
            # Check if Poetry is available
            if command -v poetry >/dev/null 2>&1; then
              echo "Poetry detected. Installing dependencies with Poetry..."
              poetry install
              echo ""
              echo "MaLO development environment ready!"
              echo "Run 'poetry shell' to activate the Poetry environment"
            else
              echo "Poetry not found. Please install Poetry first:"
              echo "curl -sSL https://install.python-poetry.org | python3 -"
              echo ""
              echo "Alternatively, you can use pip directly:"
              pip install --upgrade pip
              pip install autogen>=0.4 nyuctf>=1.1
            fi
          '';
        };
      }
    );
}