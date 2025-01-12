{
  pyproject-nix,
  python3,
}:
let
  project = pyproject-nix.lib.project.loadPyproject {
    projectRoot = ../.;
  };
  python = python3;
in
python.withPackages (project.renderers.withPackages { inherit python; })
