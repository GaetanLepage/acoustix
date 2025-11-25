{
  description = "acoustix nix flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
    in
    {
      formatter.${system} = nixpkgs.legacyPackages.${system}.nixfmt-tree;

      devShells.${system}.default =
        let
          inherit (pkgs) lib;

          stdenv' = pkgs.cudaPackages.backendStdenv;

          cudaToolkit = pkgs.symlinkJoin {
            name = "cuda";
            paths = with pkgs.cudaPackages; [
              (lib.getBin cuda_nvcc)
              (lib.getInclude cuda_cudart) # cuda_runtime.h
              (lib.getLib cuda_cudart)

              (lib.getInclude cuda_cccl) # <nv/target>

              (lib.getInclude libcurand) # libcurand.h
              (lib.getLib libcurand)

              (lib.getInclude libcufft)
              (lib.getLib libcufft)
            ];
          };
        in
        pkgs.mkShellNoCC {
          buildInputs = with pkgs; [
            uv

            # For installing gpuRIR
            stdenv'.cc
            cmake
            cudaToolkit

            # For downloading LibriSpeech
            ffmpeg
            findutils
            parallel
          ];

          QT_QPA_PLATFORM = "wayland";
          QT_PLUGIN_PATH = with pkgs.qt6; "${qtbase}/${qtbase.qtPluginPrefix}";
          # MPLBACKEND = "module://sixel";
          MPLBACKEND = "QtAgg";
          # QT_DEBUG_PLUGINS = 1;

          NIX_LD_LIBRARY_PATH = lib.makeLibraryPath (
            with pkgs;
            [
              # numpy
              libz
              stdenv'.cc.cc
              zstd

              # torchcodec (torchaudio)
              ffmpeg

              # matplotlib
              dbus
              fontconfig
              freetype
              glib
              libGL
              libx11
              libxkbcommon
              wayland

              # for soundcard
              pulseaudio

              # CUDA support on NixOS
              "/run/opengl-driver"
            ]
          );

          shellHook = ''
            source .venv/bin/activate
            export PYTHONPATH=$(pwd):$PYTHONPATH
          '';
        };
    };
}
