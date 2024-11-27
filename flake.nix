{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { flake-utils, nixpkgs, ... }: 
  flake-utils.lib.eachDefaultSystem (system: let
    pkgs = import nixpkgs { inherit system; };
  in {
    devShells.default = let
      boostOverride = pkgs.boost.override {
        enableShared = false;
        enableStatic = true;
      };
    in pkgs.mkShell {
      buildInputs = with pkgs; [
        boostOverride
        cmake
        eigen
        libgcc
        python312
        python312Packages.pybind11
        python312Packages.numpy
        python312Packages.pandas
        python312Packages.scipy
      ];

      RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;
      CPYTHON_HEADER_PATH = "${pkgs.python312}/include/python3.12";
      CPLUS_INCLUDE_PATH = "${pkgs.python312}/include/python3.12";
      BOOST_ROOT = "${boostOverride}";

      shellHook = ''
        clear; zsh; exit
      '';
    };
  });
}
