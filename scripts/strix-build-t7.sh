#!/usr/bin/env bash
# T7 build: native (host-runnable) Vulkan-only build of this fork, same flags as
# scripts/strix-build-t5b.sh, with ccache from /var/home/sleepy/.cache/ccache-strix
# (shared across agents). Builds llama-cli + llama-server (the reduced-vocab MTP
# drafter lives in common/speculative.cpp, pulled in by both via llama-common).
#
# Build dir defaults to build-vulkan-t7 (T7's own dir; never rebuilds into
# build-vulkan/, build-vulkan-t5/ or build-vulkan-t5b/, which belong to T1/T4/T5b).
#
# Usage: scripts/strix-build-t7.sh [build-dir]
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${1:-${SRC_DIR}/build-vulkan-t7}"
IMAGE="${STRIX_BUILD_IMAGE:-docker.io/library/fedora:44}"
JOBS="${STRIX_BUILD_JOBS:-16}"   # box runs other services; do not use all 32 threads
CCACHE_DIR_HOST="${STRIX_CCACHE_DIR:-/var/home/sleepy/.cache/ccache-strix}"

echo "building ${SRC_DIR} -> ${BUILD_DIR} (image ${IMAGE}, -j ${JOBS}, ccache ${CCACHE_DIR_HOST})"

mkdir -p "${BUILD_DIR}" "${CCACHE_DIR_HOST}"

podman run --rm \
    --security-opt label=disable \
    -v "${SRC_DIR}:/work:z" \
    -v "${CCACHE_DIR_HOST}:/ccache:z" \
    -w /work \
    "${IMAGE}" \
    bash -c "
        set -euo pipefail
        dnf -y -q --nodocs install gcc gcc-c++ cmake ninja-build make git \
            glslang glslc vulkan-headers vulkan-loader-devel spirv-headers-devel ccache
        export CCACHE_DIR=/ccache
        cmake -S /work -B /work/${BUILD_DIR##*/} \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            -DGGML_VULKAN=ON \
            -DGGML_NATIVE=ON \
            -DBUILD_SHARED_LIBS=OFF \
            -DLLAMA_BUILD_COMMON=ON
        # build llama-cli (model dry-run / validation) + llama-server (reduced-vocab
        # MTP drafter); a full build is:
        #   cmake --build /work/${BUILD_DIR##*/} --config Release -j ${JOBS}
        cmake --build /work/${BUILD_DIR##*/} --config Release -j ${JOBS} \
            --target llama-cli llama-server
    "

echo
echo "done. Run the host binary directly, e.g.:"
echo "  ${BUILD_DIR}/bin/llama-cli -m <model.gguf> --dry-run"
echo "  ${BUILD_DIR}/bin/llama-server --help | grep vocab-reduced"
