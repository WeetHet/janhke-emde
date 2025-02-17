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
    inputs@{ self, flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;
      imports = [
        inputs.treefmt-nix.flakeModule
        ./defs/useScipy1.15.nix
      ];

      flake = {
        project = inputs.pyproject-nix.lib.project.loadPyproject {
          projectRoot = ./.;
        };
      };

      perSystem =
        { pkgs, ... }:
        {
          packages.default = pkgs.callPackage ./defs/package.nix {
            inherit (self) project;
          };

          devShells.default = pkgs.mkShellNoCC {
            packages = [
              (pkgs.callPackage ./defs/pythonEnv.nix {
                inherit (self) project;
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

            settings = {
              excludes = [
                "*.md"
                "nix"
                "*.png"
              ];
              formatter.ruff-check.priority = 1;
              formatter.ruff-format.priority = 2;
            };
          };
        };
    };
}
