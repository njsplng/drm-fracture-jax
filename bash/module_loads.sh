#!/bin/bash

# Safeguard to ensure that this does not stop execution on error
command_exists() { command -v "$1" >/dev/null 2>&1; }

# Load the modules if on hpc
if command_exists module; then
    ml purge && module load GCC/13.2.0 libGLU/9.0.3 Python/3.11.5
fi
