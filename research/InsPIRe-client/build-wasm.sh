#!/bin/bash
# Build script for WebAssembly target
# This script builds the InsPIRe library for WebAssembly without terminal dependencies

echo "Building InsPIRe for WebAssembly..."
wasm-pack build --target web --out-dir web/pkg --no-default-features

if [ $? -eq 0 ]; then
    echo "✅ WebAssembly build successful!"
    echo "📦 Output files are in web/pkg/"
    ls -la web/pkg/
else
    echo "❌ WebAssembly build failed!"
    exit 1
fi
