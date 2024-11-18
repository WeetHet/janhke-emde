{
  pyproject-nix,
  uv2nix,
  callPackage,
}:
let
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
  pythonSet = callPackage ./nix/mkPythonSet.nix {
    inherit uv2nix pyproject-nix workspace;
  };
in
pythonSet.mkVirtualEnv "janhke-emde" {
  janhke-emde = [ ];
}
