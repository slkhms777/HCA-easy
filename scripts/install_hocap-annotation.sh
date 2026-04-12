#!/bin/bash

# Load shared config
source "$(dirname "$0")/config.sh"

# install torch (12.4)
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies from requirements.txt
log_message "Installing Python dependencies from requirements.txt..."
if "${PYTHON_PATH}" -m pip install --no-cache-dir -r "${PROJ_ROOT}/requirements.txt"; then
    log_message "Python dependencies installed successfully."
else
    handle_error "Failed to install Python dependencies."
fi

# Install chumpy from GitHub
# log_message "Installing chumpy from GitHub..."
# if "${PYTHON_PATH}" -m pip install --no-cache-dir "git+https://github.com/gobanana520/chumpy.git"; then
#     log_message "chumpy installed successfully."
# else
#     handle_error "Failed to install chumpy."
# fi

# Install manopth from GitHub
log_message "Installing manopth from GitHub..."
if "${PYTHON_PATH}" -m pip install --no-cache-dir "git+https://github.com/gobanana520/manopth.git"; then
    log_message "manopth installed successfully."
else
    handle_error "Failed to install manopth."
fi

# Install hocap_annotation
log_message "Installing hocap-annotation..."
if "${PYTHON_PATH}" -m pip install -e . --no-cache-dir; then
    log_message "hocap_annotation installed successfully."
else
    handle_error "Failed to install hocap-annotation."
fi

# Build meshsdf_loss
log_message "Building meshsdf_loss..."
MYCUDA_DIR="${PROJ_ROOT}/hocap_annotation/loss/meshsdf_loss"

# Navigate to meshsdf_loss directory
if cd "$MYCUDA_DIR"; then
    log_message "Cleaning previous build artifacts in meshsdf_loss..."
    rm -rf build *egg* *.so

    log_message "Installing meshsdf_loss..."
    if "${PYTHON_PATH}" -m pip install . --no-cache-dir --no-build-isolation; then
        log_message "meshsdf_loss installed successfully."
    else
        handle_error "Failed to install meshsdf_loss."
    fi
else
    handle_error "Failed to cd to meshsdf_loss directory: $MYCUDA_DIR"
fi

# Return to the project root directory
cd "$PROJ_ROOT" || handle_error "Failed to cd to project root directory: $PROJ_ROOT"

log_message "All build steps completed successfully."
