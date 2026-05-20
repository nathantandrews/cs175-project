#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Target directory (resolves to cs175-project/data)
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
mkdir -p "$DATA_DIR"

# Check if zstd is installed
if ! command -v zstd &> /dev/null; then
    echo "CRITICAL ERROR: 'zstd' is not installed or not in PATH."
    echo "Please install it first (e.g., 'sudo apt install zstd')."
    exit 1
fi

# Determine download tool
DOWNLOAD_CMD=""
if command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -L -o"
elif command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -O"
else
    echo "CRITICAL ERROR: Neither 'curl' nor 'wget' is installed."
    echo "Please install one of them to download the datasets."
    exit 1
fi

# List of datasets to download
LINKS=(
    "https://github.com/j-klawson/security-gym/releases/download/v0.3.3/benign_v3.db.zst"
    "https://github.com/j-klawson/security-gym/releases/download/v0.3.3/campaigns_v2.db.zst"
    "https://github.com/j-klawson/security-gym/releases/download/v0.3.3/exp01_90d.db.zst"
    "https://github.com/j-klawson/security-gym/releases/download/v0.3.3/exp_30d_heavy.db.zst"
    "https://github.com/j-klawson/security-gym/releases/download/v0.3.3/exp_7d_brute.db.zst"
    "https://github.com/j-klawson/security-gym/releases/download/v0.3.3/exp_365d_realistic.db.zst"
)

# Loop and process each link
for link in "${LINKS[@]}"; do
    filename=$(basename "$link")
    zst_path="$DATA_DIR/$filename"
    db_path="$DATA_DIR/${filename%.zst}"

    if [ -f "$db_path" ]; then
        echo "Database $(basename "$db_path") already exists, skipping."
        continue
    fi

    echo "Downloading $filename..."
    $DOWNLOAD_CMD "$zst_path" "$link"

    echo "Extracting $filename..."
    # --rm deletes the original .zst file upon successful extraction
    zstd -d --rm "$zst_path"

    echo "Successfully extracted $(basename "$db_path")"
    echo "----------------------------------------"
done

echo "All datasets processed successfully!"
