{ pkgs ? import <nixpkgs> { } }:
(import ./nix/from-sources.nix {
  inherit pkgs;
  sources = import ./npins;
  workspaceRoot = ./.;
}).pythonSet.mkVirtualEnv "janhke-emde"
{
  janhke-emde = [ ];
}
