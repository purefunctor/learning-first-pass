#!/usr/bin/env sh

SIZE=64

cargo build --release 2> /dev/null

# generate test data
./target/release/learning-first-pass --output output-test --size $SIZE 3000

# pushd output-test
# convert *.ppm image_%d.png
# popd

# generate training data
./target/release/learning-first-pass --output output-train --size $SIZE 12000

# pushd output-train
# convert *.ppm image_%d.png
# popd
