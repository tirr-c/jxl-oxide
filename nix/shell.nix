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
  fenix ? import (getFlake "fenix") { },
  rustVersion ? "stable",
  ...
}:

let
  pkgsFromNixpkgs = with pkgs; [
    ffmpeg_7-headless
    pkg-config

    rustPlatform.bindgenHook
  ];

  # Copied from naersk
  pkgsDarwin = with pkgs.darwin; [
    Security
    apple_sdk.frameworks.CoreServices
    apple_sdk.frameworks.SystemConfiguration
    cf-private
    libiconv
  ];

  rustToolchain = fenix.${rustVersion}.withComponents [
    "cargo"
    "clippy"
    "rust-analyzer"
    "rust-src"
    "rustc"
    "rustfmt"
  ];
in

pkgs.mkShell {
  name = "jxl-oxide";
  nativeBuildInputs =
    pkgsFromNixpkgs ++ pkgs.lib.optionals pkgs.stdenv.isDarwin pkgsDarwin ++ [ rustToolchain ];
}
