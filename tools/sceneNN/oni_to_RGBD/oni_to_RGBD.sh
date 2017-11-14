#!/bin/bash
for filename in $1*.oni; do
    directory=$2${filename: -7:-4}
    if [ ! -d $directory ]; then
      mkdir $directory
    fi
    echo $2${filename: -7:-4}
    ./extractRGBD $filename $2${filename: -7:-4}
    rm $filename
done
