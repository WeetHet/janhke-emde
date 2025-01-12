{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;
      imports = [ inputs.treefmt-nix.flakeModule ];

      perSystem =
        { pkgs, ... }:
        {
          packages.default = pkgs.callPackage ./defs/package.nix {
            inherit (inputs) pyproject-nix;
          };

          devShells.default = pkgs.mkShellNoCC {
            packages = [
              (import ./defs/pythonEnv.nix {
                inherit (pkgs) python3;
                inherit (inputs) pyproject-nix;
              })
              pkgs.ruff
            ];
          };

          treefmt = {
            projectRootFile = "flake.nix";
            programs.ruff = {
              check = true;
              format = true;
            };
            programs.nixfmt.enable = true;
            programs.shfmt.enable = true;
            programs.taplo.enable = true;

            settings.formatter = {
              ruff-check.priority = 1;
              ruff-format.priority = 2;
            };
          };
        };
    };
}
