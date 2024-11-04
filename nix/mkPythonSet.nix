{ uv2nix
, pyproject-nix
, python3
, callPackage
, lib
, workspaceRoot
}:
let
  workspace = uv2nix.lib.workspace.loadWorkspace { inherit workspaceRoot; };
  overlay = import ./overlay.nix { inherit lib workspace; };
  pythonSet = (callPackage pyproject-nix.build.packages {
    python = python3;
  }).overrideScope overlay;
in
{
  inherit pythonSet workspace;
}
