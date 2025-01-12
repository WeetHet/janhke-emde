{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flakelight = {
      url = "github:nix-community/flakelight";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs =
    { flakelight, pyproject-nix, ... }:
    flakelight ./. (
      { lib, ... }:
      {
        systems = lib.systems.flakeExposed;
        package = { callPackage }: callPackage ./defs/package.nix { inherit pyproject-nix; };
        devShell.packages = pkgs: [
          (import ./defs/pythonEnv.nix {
            inherit (pkgs) python3;
            inherit pyproject-nix;
          })
          pkgs.ruff
        ];
        formatter = pkgs: pkgs.nixfmt-rfc-style;
      }
    );
}
