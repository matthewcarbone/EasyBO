#!/bin/bash

pip install toml
for var in "$@"; do
    if [[ "$var" == "dev" ]]; then
        python3 -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["optional-dependencies"]["dev"]))' | pip install -r /dev/stdin
    elif [[ "$var" == "docs" ]]; then
        python3 -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["optional-dependencies"]["docs"]))' | pip install -r /dev/stdin
    elif [[ "$var" == "extra" ]]; then
        python3 -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["optional-dependencies"]["extra"]))' | pip install -r /dev/stdin
    elif [[ "$var" == "core" ]]; then
        python3 -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["dependencies"]))' | pip install -r /dev/stdin
    fi
done
