{ pkgs, sources, workspaceRoot }:
let
  inherit (pkgs) lib;
  uv2nix = (import ./flake-compat.nix { src = sources.uv2nix; }).outputs;
  pyproject-nix = import sources.pyproject-nix { inherit lib; };
in
import ./mkPythonSet.nix {
  inherit (pkgs) python3 callPackage;
  inherit lib uv2nix pyproject-nix workspaceRoot;
}
