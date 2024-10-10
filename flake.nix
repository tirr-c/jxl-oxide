{
  inputs = {
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
    naersk = {
      url = "github:nix-community/naersk";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      fenix,
      flake-utils,
      naersk,
      nixpkgs,
    }:
    let
      crossTargets = [
        {
          target = "x86_64-unknown-linux-gnu";
          isStatic = false;
        }
        {
          target = "x86_64-unknown-linux-musl";
          isStatic = true;
        }
        {
          target = "x86_64-pc-windows-gnu";
          isStatic = false;
          nixSystem = "x86_64-w64-mingw32";
        }
        {
          target = "aarch64-unknown-linux-gnu";
          isStatic = false;
        }
        {
          target = "aarch64-unknown-linux-musl";
          isStatic = true;
        }
        {
          target = "armv7-unknown-linux-gnueabihf";
          isStatic = false;
          nixSystem = "armv7l-unknown-linux-gnueabihf";
        }
        {
          target = "armv7-unknown-linux-musleabihf";
          isStatic = true;
          nixSystem = "armv7l-unknown-linux-musleabihf";
        }
      ];
    in
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        pkgsCrossFor =
          target:
          import nixpkgs {
            localSystem = system;
            crossSystem = {
              config = target;
            };
          };
        inherit (pkgs) lib;
        inherit (builtins)
          map
          listToAttrs
          replaceStrings
          isNull
          ;

        mapListToAttrs = f: l: listToAttrs (map f l);
        toScreamingSnakeCase = s: replaceStrings [ "-" ] [ "_" ] (lib.strings.toUpper s);

        rustVersion = "stable";
        toolchainBase = fenix.packages.${system};
        toolchainFor =
          target:
          with toolchainBase;
          combine (
            [
              toolchainBase.${rustVersion}.rustc
              toolchainBase.${rustVersion}.cargo
            ]
            ++ (lib.optional (!isNull target) targets.${target}.${rustVersion}.rust-std)
          );
        naerskFor =
          target:
          naersk.lib.${system}.override {
            cargo = toolchainFor target;
            rustc = toolchainFor target;
          };
        naerskForNative = naerskFor null;

        commonBuildArgs = {
          pname = "jxl-oxide";
          src = ./.;
          strictDeps = true;
          overrideMain = old: {
            preConfigure = ''
              cargo_build_options="$cargo_build_options -p jxl-oxide-cli"
            '';
          };
        };

        naerskBuildPackageNative =
          let
            isMinGW = pkgs.stdenv.cc.isGNU or false && pkgs.hostPlatform.isWindows;
          in
          {
            useClang ? isMinGW,
            ...
          }@naerskArgs:
          let
            stdenv = if useClang then pkgs.clangStdenv else pkgs.stdenv;
          in
          naerskForNative.buildPackage (
            commonBuildArgs
            // naerskArgs
            // {
              depsBuildBuild = [ stdenv.cc ];
            }
          );

        naerskBuildPackageCross =
          {
            target,
            isStatic,
            nixSystem ? target,
          }:
          let
            pkgsCross = pkgsCrossFor nixSystem;
            inherit (pkgsCross) hostPlatform;
            targetForCargoEnv = toScreamingSnakeCase target;
            isMinGW = pkgsCross.stdenv.cc.isGNU or false && hostPlatform.isWindows;
            naersk' = naerskFor target;
          in
          {
            useClang ? isMinGW,
            ...
          }@naerskArgs:
          let
            stdenv = if useClang then pkgsCross.clangStdenv else pkgsCross.stdenv;
          in
          naersk'.buildPackage (
            commonBuildArgs
            // naerskArgs
            // rec {
              depsBuildBuild = [
                stdenv.cc
              ] ++ lib.optional isMinGW pkgsCross.windows.pthreads;

              CARGO_BUILD_TARGET = target;
              TARGET_CC = "${stdenv.cc}/bin/${stdenv.cc.targetPrefix}cc";
              "CARGO_TARGET_${targetForCargoEnv}_LINKER" = TARGET_CC;
            }
            // lib.optionalAttrs isStatic {
              "CARGO_TARGET_${targetForCargoEnv}_RUSTFLAGS" = "-C target-feature=+crt-static";
            }
          );

        crossPackages = mapListToAttrs (
          spec@{ target, ... }:
          {
            name = target;
            value = naerskBuildPackageCross spec { };
          }
        ) crossTargets;
      in
      rec {
        packages = {
          native = naerskBuildPackageNative { };
          native-devtools = naerskBuildPackageNative {
            cargoBuildOptions =
              args:
              args
              ++ [
                "--features"
                "__devtools"
              ];
          };
        } // crossPackages;
        defaultPackage = packages.native;

        devShell =
          let
            pkgsFromNixpkgs = with pkgs; [
              ffmpeg_7-headless
              pkg-config

              rustPlatform.bindgenHook
            ];
            rustToolchain = toolchainBase.${rustVersion}.withComponents [
              "cargo"
              "clippy"
              "rust-analyzer"
              "rust-src"
              "rustc"
              "rustfmt"
            ];
          in
          pkgs.mkShell {
            nativeBuildInputs = pkgsFromNixpkgs ++ [ rustToolchain ];
          };

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
