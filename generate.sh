#!/usr/bin/env sh

# generate test data
cargo run -q --release -- --output-path output-test 0 3000

pushd output-test
convert *.ppm image_%d.png
popd

# generate training data
cargo run -q --release -- --output-path output-train 0 12000

pushd output-train
convert *.ppm image_%d.png
popd
