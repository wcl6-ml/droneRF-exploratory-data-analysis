#!/bin/bash

# Target specific folder or default
DATA_DIR="${1:-./data/raw/DroneRF}"

echo "Processing: $DATA_DIR"

find "$DATA_DIR" -name "*.rar" -type f | while read -r rar_file; do
    parent_dir=$(dirname "$rar_file")
    folder_name=$(basename "$rar_file" .rar)
    dest_path="$parent_dir/$folder_name"

    # 1. Create the clean destination
    mkdir -p "$dest_path"
    
    echo "Extracting $folder_name..."

    # 2. 'e' extracts files to the root of dest_path, 
    # ignoring any internal folder structure in the .rar
    unrar e -o+ -idq "$rar_file" "$dest_path/"
    
    # 3. Cleanup: If unrar created an empty nested folder (it shouldn't with 'e'), 
    # but some versions of unrar behave differently, we ensure only files remain.
    find "$dest_path" -type d -empty -delete
done

echo "Clean extraction complete."