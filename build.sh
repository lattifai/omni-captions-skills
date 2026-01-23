#!/bin/bash
set -e

# Get latest version tar.gz from a directory
latest_pkg() {
    ls -1 "$1"/*.tar.gz 2>/dev/null | sort -V | tail -1
}

# Build sibling projects
(cd ../lattifai-core && ./build.sh)
(cd ../lattifai-captions && python -m build --sdist)
(cd ../lattifai-python && python -m build --sdist)

# Build this package
python -m build --sdist

# Collect latest version from all projects
rm -rf packages && mkdir -p packages
cp "$(latest_pkg dist)" packages/
cp "$(latest_pkg ../lattifai-core/dist)" packages/
cp "$(latest_pkg ../lattifai-captions/dist)" packages/
cp "$(latest_pkg ../lattifai-python/dist)" packages/

ls -la packages/*.tar.gz
