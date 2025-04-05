let
  inherit (builtins) fetchTarball fromJSON readFile;
  getFlake =
    name: with (fromJSON (readFile ./flake.lock)).nodes.${name}.locked; {
      inherit rev;
      outPath = fetchTarball {
        url = "https://github.com/${owner}/${repo}/archive/${rev}.tar.gz";
        sha256 = narHash;
      };
    };
in

{
  system ? builtins.currentSystem,
  pkgs ? import (getFlake "nixpkgs") {
    localSystem = {
      inherit system;
    };
  },
  fenix ? (import (getFlake "fenix") { }).packages.${system},
  toolchainSpec,
  ...
}:

let
  pkgsFromNixpkgs = with pkgs; [
    ffmpeg_7-headless
    nodejs
    pkg-config
    wasm-pack
    wasm-bindgen-cli

    rustPlatform.bindgenHook
  ];

  # Copied from naersk
  pkgsDarwin = with pkgs.darwin; [
    Security
    apple_sdk.frameworks.CoreFoundation
    apple_sdk.frameworks.CoreServices
    apple_sdk.frameworks.SystemConfiguration
    libiconv
  ];

  rustToolchain = fenix.combine (
    with fenix.toolchainOf toolchainSpec;
    [
      cargo
      clippy
      rust-analyzer
      rust-src
      rustc
      rustfmt
    ]
    ++ [
      (fenix.targets.wasm32-unknown-unknown.toolchainOf toolchainSpec).rust-std
    ]
  );
in

pkgs.mkShell {
  name = "jxl-oxide";
  nativeBuildInputs =
    pkgsFromNixpkgs ++ pkgs.lib.optionals pkgs.stdenv.isDarwin pkgsDarwin ++ [ rustToolchain ];
}
