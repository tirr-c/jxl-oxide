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

  pkgs = import <nixpkgs> { };
  fenix = pkgs.callPackage (getFlake "fenix") { };
  toolchain = fenix.stable.withComponents [
    "cargo"
    "rustc"
  ];
  naersk = pkgs.callPackage (getFlake "naersk") {
    cargo = toolchain;
    rustc = toolchain;
  };
in

pkgs.callPackage ./nix/build.nix {
  inherit naersk;
}
