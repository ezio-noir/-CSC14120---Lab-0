#!/bin/bash

sizes=(64 256 1024 4096 16384 65536 262144 1048576 4194304 16777216)

echo "-------- Perform vector addition on host (version 0) --------"
for size in "${sizes[@]}"; do
    ./P2 -n "$size" -v 0 -f "./data/$size"
done

echo "-------- Perform vector addition on device, version 1 -------"
for size in "${sizes[@]}"; do
    ./P2 -n "$size" -v 1 -f "./data/$size"
done

echo "-------- Perform vector addition on device, version 2 -------"
for size in "${sizes[@]}"; do
    ./P2 -n "$size" -v 2 -f "./data/$size"
done
