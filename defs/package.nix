{
  pyproject-nix,
  python3,
}:
let
  project = pyproject-nix.lib.project.loadPyproject {
    projectRoot = ../.;
  };
  python = python3;
  attrs = project.renderers.buildPythonPackage { inherit python; };
in
python.pkgs.buildPythonPackage attrs
