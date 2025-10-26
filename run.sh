#!/bin/bash

export UV_VENV_CLEAR=1

function log(){
    echo $1 
}

function download_verdict(){
    local url="https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    local filename="the-verdict.txt"

    # If file exists, delete it
    if [ -f "$filename" ]; then
        echo "Removing existing $filename..."
        rm -f "$filename"
    fi

    # Download the file
    echo "Downloading $filename..."
    curl -L -o "$filename" "$url"
}

case $1 in
    st)
        log "Setup env" 
        uv venv --python 3.11
        source .venv/bin/activate
        uv pip install -r ./requirements.txt
        ;;
    ds)
        log "Downloading Dataset"
        download_verdict
        ;;
    rj)
        log "Running Jupiterlab"
        uv run jupyter lab
        ;;
    rp)
        uv run $2
        ;;
    rt)
        uv run pytest -vvs $2
        ;;
    *)
        log "Invalid option. Please choose 1, 2, or 3."
        ;;
esac


