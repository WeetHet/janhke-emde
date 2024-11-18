{
  pyproject-nix,
  uv2nix,
  devshell,
  uv,
  callPackage,
}:
let
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
  pythonSet = callPackage ./nix/mkPythonSet.nix {
    inherit pyproject-nix workspace;
  };
  editableOverlay = workspace.mkEditablePyprojectOverlay {
    root = "$REPO_ROOT";
  };
  editablePythonSet = pythonSet.overrideScope editableOverlay;
  virtualenv = editablePythonSet.mkVirtualEnv "janhke-emde-dev-env" {
    janhke-emde = [ ];
  };
in
devshell.mkShell {
  packages = [
    virtualenv
    uv
  ];
  env = [
    {
      name = "PYTHONPATH";
      unset = true;
    }
    {
      name = "REPO_ROOT";
      eval = "$(git rev-parse --show-toplevel)";
    }
  ];
  commands = [
    {
      help = "add a dependency cleaning up afterwards";
      name = "uv-add";
      command = "uv add $1 && rm -rf .venv";
    }
  ];
}
