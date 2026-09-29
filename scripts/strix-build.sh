#!/usr/bin/env bash
# Native (host-runnable) Vulkan-only build of this fork for the Strix Halo box.
#
# Why a container: the host is Bazzite (rpm-ostree, immutable) and lacks
# vulkan-headers, glslang and glslc; we do not layer the host. The container
# only provides the build toolchain. The source and the build tree live on a
# bind-mounted workspace volume, and fedora:44 and the host share the same
# glibc (2.43-8.fc44), so the resulting binaries run directly on the host.
#
# Usage: scripts/strix-build.sh [build-dir]
#   build-dir defaults to ./build-vulkan (ignored by .gitignore via build*)
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${1:-${SRC_DIR}/build-vulkan}"
IMAGE="${STRIX_BUILD_IMAGE:-docker.io/library/fedora:44}"
JOBS="${STRIX_BUILD_JOBS:-16}"   # box runs other services; do not use all 32 threads

# Fedora splits the Vulkan build deps differently than Ubuntu:
#  - glslc is its own package (Ubuntu bundles it in the "glslc" equivalent of
#    spirv-tools; the old ~/.llama/vulkan-llama Dockerfile uses Ubuntu names)
#  - vulkan-loader-devel provides libvulkan.so for the linker; the *runtime*
#    loader comes from the host (/usr/lib64/libvulkan.so.1, 1.4.341)
#  - spirv-headers-devel (not "spirv-headers") provides the CMake package
echo "building ${SRC_DIR} -> ${BUILD_DIR} (image ${IMAGE}, -j ${JOBS})"

mkdir -p "${BUILD_DIR}"

podman run --rm \
    --security-opt label=disable \
    -v "${SRC_DIR}:/work:z" \
    -w /work \
    "${IMAGE}" \
    bash -c "
        set -euo pipefail
        dnf -y -q --nodocs install gcc gcc-c++ cmake ninja-build make git \
            glslang glslc vulkan-headers vulkan-loader-devel spirv-headers-devel
        cmake -S /work -B /work/${BUILD_DIR##*/} \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
            -DGGML_VULKAN=ON \
            -DGGML_NATIVE=ON \
            -DBUILD_SHARED_LIBS=OFF
        cmake --build /work/${BUILD_DIR##*/} --config Release -j ${JOBS}
    "

echo
echo "done. Run the host binary directly, e.g.:"
echo "  ${BUILD_DIR}/bin/llama-bench --list-devices"
echo
echo "Serving this model (shard 2 auto-discovered, PLE streamed from disk):"
echo "  ${BUILD_DIR}/bin/llama-server \\"
echo "    -m <model-00001.gguf> --n-gpu-layers 999 --device vulkan0 \\"
echo "    --ngram-on-disk --ngram-io-threads 64 --ngram-cache 256"
