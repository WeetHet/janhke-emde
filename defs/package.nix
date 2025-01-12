{
  python3,
  project,
}:
python3.pkgs.buildPythonPackage (project.renderers.buildPythonPackage { python = python3; })
