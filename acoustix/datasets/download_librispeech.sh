#!/bin/sh

set -e

ARCHIVE_NAME="train-clean-100.tar.gz"
LIBRISPEECH_URL="https://openslr.trmal.net/resources/12/$ARCHIVE_NAME"
DEST_PATH="data/LibriSpeech"

if [ ! -d "$DEST_PATH" ]; then
    echo "Downloading $ARCHIVE_NAME"
    mkdir -p "data/"
    curl -o "data/$ARCHIVE_NAME" "$LIBRISPEECH_URL"
    echo "Extracting $ARCHIVE_NAME to $DEST_PATH"
    tar zxvf "data/$ARCHIVE_NAME" --directory "data/"
fi

n_cpus=$(getconf _NPROCESSORS_ONLN)
ffmpeg_command="[ ! -e {.}.wav ] && ffmpeg -loglevel quiet -n -i {} {.}.wav"

echo "Converting all .flac files to .wav"
find "$DEST_PATH" -type f -name "*.flac" |
    parallel --will-cite -j "$n_cpus" "$ffmpeg_command"

echo "Finished!"
