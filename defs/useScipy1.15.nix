{ inputs, ... }:
{
  perSystem =
    { system, ... }:
    {
      _module.args.pkgs = import inputs.nixpkgs {
        inherit system;
        overlays = [
          (self: super: {
            python3 = super.python3.override {
              packageOverrides = python-self: python-super: {
                scipy = python-super.scipy.overrideAttrs (attrs: rec {
                  version = "1.15.1";
                  src = super.fetchFromGitHub {
                    owner = "scipy";
                    repo = "scipy";
                    tag = "v${version}";
                    hash = "sha256-pQfOiK/i/Nz1mCGdDA7ivnzHxqee1WVD62CxxgetGLg=";
                    fetchSubmodules = true;
                  };

                  postPatch = null;
                  doCheck = false;
                  pytestCheckPhase = ''true'';
                });
              };
            };
          })
        ];
      };
    };
}
