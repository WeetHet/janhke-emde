{
  pyproject-nix,
  python3,
  callPackage,
  lib,
  workspace,
}:
let
  overlay = import ./overlay.nix { inherit lib workspace; };
in
(callPackage pyproject-nix.build.packages {
  python = python3;
}).overrideScope
  overlay
