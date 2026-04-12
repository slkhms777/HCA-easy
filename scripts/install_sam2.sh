#!/bin/bash

# Load shared config
source "$(dirname "$0")/config.sh"

# Install Python dependencies from requirements_sam2.txt
log_message "Installing Python dependencies from requirements_sam2.txt..."
if "${PYTHON_PATH}" -m pip install --no-cache-dir -r "${PROJ_ROOT}/requirements_sam2.txt"; then
    log_message "Python dependencies installed successfully."
else
    handle_error "Failed to install Python dependencies."
fi

# Build SAM 2 CUDA extension
export SAM2_BUILD_CUDA=1
export SAM2_BUILD_ALLOW_ERRORS=0

# Build SAM2
SAM2_DIR="${PROJ_ROOT}/third_party/sam2"
log_message "Building SAM2..."

# Initialize the SAM2 submodule
log_message "Initializing SAM2 submodule..."
if git submodule update --init --recursive -- "${SAM2_DIR}"; then
    log_message "SAM2 submodule initialized successfully."
else
    handle_error "Failed to initialize SAM2 submodule."
fi

if cd "$SAM2_DIR"; then
    log_message "Cleaning previous builds in SAM2 directory..."
    rm -rf build *egg* *.so

    # Install SAM2 package
    log_message "Installing SAM2 package..."
    if "${PYTHON_PATH}" -m pip install . --no-build-isolation --no-cache-dir; then
        log_message "SAM2 installed successfully."
    else
        handle_error "Failed to install SAM2."
    fi

    # # Build SAM2 extension
    # log_message "Building SAM2 extension with build_ext --inplace..."
    # if "${PYTHON_PATH}" setup.py build_ext --inplace; then
    #     log_message "SAM2 extension built successfully."
    # else
    #     handle_error "Failed to build SAM2 extension."
    # fi
else
    handle_error "Failed to cd to SAM2 directory: $SAM2_DIR"
fi

# Return to project root
cd "$PROJ_ROOT" || handle_error "Failed to cd to project root directory: $PROJ_ROOT"

log_message "All build steps completed successfully."
