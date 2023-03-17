#!/usr/bin/env sh

# generate test data
cargo run -q --release -- --output-path output-test 0 200

# generate training data
cargo run -q --release -- --output-path output-train 0 800

pushd output-test
convert *.ppm image_%d.png
popd

pushd output-train
convert *.ppm image_%d.png
popd
