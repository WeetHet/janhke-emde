{ pkgs ? import <nixpkgs> { } }:
let
  inherit (import ./nix/from-sources.nix {
    inherit pkgs;
    sources = import ./npins;
    workspaceRoot = ./.;
  }) workspace pythonSet;
  editableOverlay = workspace.mkEditablePyprojectOverlay {
    root = "$REPO_ROOT";
  };
  editablePythonSet = pythonSet.overrideScope editableOverlay;
  virtualenv = editablePythonSet.mkVirtualEnv "janhke-emde-dev-env" {
    janhke-emde = [ ];
  };
in
pkgs.mkShell {
  packages = [
    virtualenv
    pkgs.uv
  ];
  shellHook = ''
    unset PYTHONPATH
    export REPO_ROOT=$(git rev-parse --show-toplevel)
  '';
}
