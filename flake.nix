{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flakelight = {
      url = "github:nix-community/flakelight";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = { flakelight, ... }:
    flakelight ./. {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      devShell.packages = pkgs: [
        (pkgs.python3.withPackages (ps: [ ps.pyvista ps.scipy ps.networkx ps.numba ]))
      ];
      formatter = pkgs: pkgs.nixfmt-rfc-style;
    };
}
