{ pkgs, sources, workspaceRoot }:
let
  inherit (pkgs) lib;
  uv2nix = (import sources.flake-compat { src = sources.uv2nix; }).defaultNix;
  pyproject-nix = import sources.pyproject-nix { inherit lib; };
in
import ./mkPythonSet.nix {
  inherit (pkgs) python3 callPackage;
  inherit lib uv2nix pyproject-nix workspaceRoot;
}
