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
  pkgs,
  lib ? pkgs.lib,
  windows ? pkgs.windows,
  hostPlatform ? pkgs.hostPlatform,
  stdenv ? pkgs.stdenv,
  naersk,

  crossTarget ? null,
  static ? false,
  enableMimalloc ? true,
  enableRayon ? true,
  enableFfmpeg ? false,
  enableDevtools ? enableFfmpeg,
  ...
}:

let
  inherit (builtins)
    concatStringsSep
    replaceStrings
    isNull
    ;

  jxlOxideCliToml = builtins.fromTOML (builtins.readFile ./crates/jxl-oxide-cli/Cargo.toml);

  toScreamingSnakeCase = s: replaceStrings [ "-" ] [ "_" ] (lib.strings.toUpper s);
  cargoEnvPrefix = "CARGO_TARGET_${toScreamingSnakeCase crossTarget}_";

  commonBuildArgs = {
    inherit (jxlOxideCliToml.package) name version;
    pname = "jxl-oxide";
    src = ./.;
    strictDeps = true;
    overrideMain = old: {
      preConfigure = ''
        cargo_build_options="$cargo_build_options -p jxl-oxide-cli"
      '';
    };
  };

  cargoBuildArgs =
    let
      featureList =
        [ ]
        ++ lib.optional enableMimalloc "mimalloc"
        ++ lib.optional enableRayon "rayon"
        ++ lib.optional enableDevtools "__devtools"
        ++ lib.optional enableFfmpeg "__ffmpeg";
      featureListStr = concatStringsSep "," featureList;
    in
    [ "--no-default-features" ]
    ++ lib.optionals (featureList != [ ]) [
      "--features"
      featureListStr
    ];

  isMinGW = stdenv.cc.isGNU or false && hostPlatform.isWindows;
in
naersk.buildPackage (
  commonBuildArgs
  // {
    cargoBuildOptions = args: args ++ cargoBuildArgs;
  }
  // lib.optionalAttrs (!isNull crossTarget) rec {
    depsBuildBuild = [
      stdenv.cc
    ] ++ lib.optional isMinGW windows.pthreads;

    CARGO_BUILD_TARGET = crossTarget;
    TARGET_CC = "${stdenv.cc}/bin/${stdenv.cc.targetPrefix}cc";
    "${cargoEnvPrefix}LINKER" = TARGET_CC;
  }
  // lib.optionalAttrs static {
    "${cargoEnvPrefix}RUSTFLAGS" = "-C target-feature=+crt-static";
  }
)
