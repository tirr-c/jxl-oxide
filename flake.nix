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
          static = false;
        }
        {
          target = "x86_64-unknown-linux-musl";
          static = true;
        }
        {
          target = "x86_64-pc-windows-gnu";
          static = false;
          nixSystem = "x86_64-w64-mingw32";
        }
        {
          target = "aarch64-unknown-linux-gnu";
          static = false;
        }
        {
          target = "aarch64-unknown-linux-musl";
          static = true;
        }
        {
          target = "armv7-unknown-linux-gnueabihf";
          static = false;
          nixSystem = "armv7l-unknown-linux-gnueabihf";
        }
        {
          target = "armv7-unknown-linux-musleabihf";
          static = true;
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
          ;

        mapListToAttrs = f: l: listToAttrs (map f l);

        rustToolchainSpec = {
          channel = "1.89.0";
          sha256 = "sha256-+9FmLhAOezBZCOziO0Qct1NOrfpjNsXxc/8I0c7BdKE=";
        };

        toolchainFor =
          target:
          with fenix.packages.${system};
          let
            toolchain = toolchainOf rustToolchainSpec;
          in
          combine (
            [
              toolchain.rustc
              toolchain.cargo
            ]
            ++ (lib.optional (target != null) (targets.${target}.toolchainOf rustToolchainSpec).rust-std)
          );
        naerskFor =
          target:
          naersk.lib.${system}.override {
            cargo = toolchainFor target;
            rustc = toolchainFor target;
          };
        naerskForNative = naerskFor null;

        # Hack to avoid mcfgthreads dependency.
        wrapMingwStdenv =
          stdenv:
          pkgs.overrideCC stdenv (
            stdenv.cc.override (old: {
              cc = old.cc.override {
                threadsCross = {
                  model = "win32";
                  package = null;
                };
              };
            })
          );

        buildPackage = pkgs.callPackage ./nix/build.nix;

        naerskBuildPackageNative =
          extraArgs:
          buildPackage (
            {
              naersk = naerskForNative;
            }
            // extraArgs
          );

        naerskBuildPackageCross =
          {
            target,
            static,
            nixSystem ? target,
            args ? { },
          }:
          let
            pkgsCross = pkgsCrossFor nixSystem;
            inherit (pkgsCross) hostPlatform stdenv;
            isMinGW = pkgsCross.stdenv.cc.isGNU or false && hostPlatform.isWindows;
          in
          buildPackage (
            {
              pkgs = pkgsCross;
              naersk = naerskFor target;
              stdenv = if isMinGW then wrapMingwStdenv stdenv else stdenv;
              crossTarget = target;
              inherit static;
            }
            // args
          );

        crossPackages = mapListToAttrs (
          spec@{ target, ... }:
          {
            name = target;
            value = naerskBuildPackageCross spec;
          }
        ) crossTargets;
      in
      rec {
        packages = {
          native = naerskBuildPackageNative { };
          native-devtools = naerskBuildPackageNative {
            enableDevtools = true;
          };
          native-ffmpeg = naerskBuildPackageNative {
            enableDevtools = true;
            enableFfmpeg = true;
          };
        } // crossPackages;
        defaultPackage = packages.native;

        devShell = pkgs.callPackage ./nix/shell.nix {
          fenix = fenix.packages.${system};
          toolchainSpec = rustToolchainSpec;
        };

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
