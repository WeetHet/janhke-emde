{ pkgs ? import <nixpkgs> { } }:
(import ./from-sources.nix {
  inherit pkgs;
  sources = import ./npins;
}).pythonSet.mkVirtualEnv "janhke-emde"
{
  janhke-emde = [ ];
}
