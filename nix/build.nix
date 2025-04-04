{
  pkgs,
  lib ? pkgs.lib,
  darwin ? pkgs.darwin,
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
    ;

  jxlOxideCliToml = builtins.fromTOML (builtins.readFile ../crates/jxl-oxide-cli/Cargo.toml);

  toScreamingSnakeCase = s: replaceStrings [ "-" ] [ "_" ] (lib.strings.toUpper s);
  cargoEnvPrefix = "CARGO_TARGET_${toScreamingSnakeCase crossTarget}_";

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
    [
      "-p"
      "jxl-oxide-cli"
      "--no-default-features"
    ]
    ++ lib.optionals (featureList != [ ]) [
      "--features"
      featureListStr
    ];

  conformance = pkgs.callPackage ./conformance.nix { };

  cargoTestArgs =
    let
      featureList =
        [ "conformance" ] # conformance tests only
        ++ lib.optional enableMimalloc "mimalloc"
        ++ lib.optional enableRayon "rayon";
      featureListStr = concatStringsSep "," featureList;
    in
    [
      "-p"
      "jxl-oxide-tests"
      "--no-default-features"
    ]
    ++ lib.optionals (featureList != [ ]) [
      "--features"
      featureListStr
    ];

  commonBuildArgs = {
    inherit (jxlOxideCliToml.package) name version;
    pname = "jxl-oxide";
    src = ../.;
    strictDeps = true;

    cargoBuildOptions = args: args ++ cargoBuildArgs;
    buildInputs =
      lib.optionals enableFfmpeg (
        with pkgs;
        [
          ffmpeg_7-headless
          rustPlatform.bindgenHook
        ]
      )
      ++ lib.optionals stdenv.isDarwin [ darwin.apple_sdk.frameworks.SystemConfiguration ];
    nativeBuildInputs = lib.optionals enableFfmpeg (
      with pkgs;
      [
        pkg-config
      ]
    );

    doCheck = true;
    cargoTestOptions = args: args ++ cargoTestArgs;
    checkInputs = [ conformance ];
    preCheck = ''
      export JXL_OXIDE_CACHE=${conformance}/cache
      export JXL_OXIDE_CONFORMANCE_TESTCASES=${conformance}/testcases
    '';
    postCheck = ''
      unset JXL_OXIDE_CACHE
      unset JXL_OXIDE_CONFORMANCE_TESTCASES
    '';
  };

  isMinGW = stdenv.cc.isGNU or false && hostPlatform.isWindows;
in
naersk.buildPackage (
  commonBuildArgs
  // lib.optionalAttrs (crossTarget != null) rec {
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
