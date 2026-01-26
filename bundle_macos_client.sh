#!/bin/bash

set -e

BUILD_TYPE="${1:-debug}"

if [ "$BUILD_TYPE" != "debug" ] && [ "$BUILD_TYPE" != "release" ]; then
    echo "Usage: $0 [debug|release]"
    exit 1
fi

CARGO_FLAGS=""
if [ "$BUILD_TYPE" = "release" ]; then
    CARGO_FLAGS="--release"
fi

echo "Building project ($BUILD_TYPE)..."
cargo build $CARGO_FLAGS

PROJECT_NAME="project-107cf1"
BUNDLE_DIR="target/$BUILD_TYPE/bundle/$PROJECT_NAME.app"
CONTENTS_DIR="$BUNDLE_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
FRAMEWORKS_DIR="$CONTENTS_DIR/Frameworks"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

echo "Creating bundle structure..."
mkdir -p "$MACOS_DIR"
mkdir -p "$FRAMEWORKS_DIR"
mkdir -p "$RESOURCES_DIR"

echo "Copying binary..."
cp target/$BUILD_TYPE/$PROJECT_NAME "$MACOS_DIR/"

echo "Copying resources..."
if [ -f "target/$BUILD_TYPE/resources" ]; then
    cp target/$BUILD_TYPE/resources "$RESOURCES_DIR/"
fi

echo "Copying libvulkan.dylib..."
if [ -z "$VULKAN_SDK" ]; then
    echo "Error: VULKAN_SDK environment variable not set"
    exit 1
fi

VULKAN_LIB=$(find "$VULKAN_SDK/lib" -name "libvulkan.*.dylib" | head -n 1)
if [ -z "$VULKAN_LIB" ]; then
    echo "Error: libvulkan.*.dylib not found in $VULKAN_SDK/lib"
    exit 1
fi

cp "$VULKAN_LIB" "$FRAMEWORKS_DIR/libvulkan.dylib"

echo "Copying Info.plist..."
cp client/Info.plist "$CONTENTS_DIR/"

echo "Bundle created at $BUNDLE_DIR"
