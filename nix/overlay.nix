{ lib, workspace, ... }:
let
  overlay' = workspace.mkPyprojectOverlay {
    sourcePreference = "sdist";
  };
  # - https://adisbladis.github.io/uv2nix/FAQ.html
  # - https://github.com/nix-community/poetry2nix/blob/master/docs/edgecases.md
  overrides = self: super: { };
in
lib.composeExtensions overlay' overrides
