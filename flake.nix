{
  description = "Raytracing, but with Machine Learning";

  inputs = {
    nixpkgs.url =
      "github:nixos/nixpkgs/22.11";

    rust-overlay.url = "github:oxalica/rust-overlay";

    flake-utils.url = "github:numtide/flake-utils";

    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, ... }:
    let supportedSystems = [ "x86_64-linux" "x86_64-darwin" ];
    in flake-utils.lib.eachSystem supportedSystems (system:
      let
        pkgs = import nixpkgs { inherit system; overlays = [ rust-overlay.overlays.default ]; };
      in
      {
        devShells = {
          default = pkgs.mkShell {
            name = "learning-first-pass";
            packages = [
              pkgs.rust-bin.stable.latest.complete
            ];
          };
        };
        formatter = pkgs.nixpkgs-fmt;
      });
}
