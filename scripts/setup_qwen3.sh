#!/bin/bash
# Setup script for Qwen3-0.6B model

set -e

# Configuration
MODEL_DIR="models/Qwen3-0.6B"
REPO_URL="https://github.com/lsb/Qwen3-0.6B.git"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Qwen3-0.6B Model Setup ==="
echo

# Check if model already exists
if [ -f "$MODEL_DIR/model.safetensors" ]; then
    echo -e "${YELLOW}Model already exists at $MODEL_DIR/model.safetensors${NC}"
    echo "Skipping setup. Delete the file to re-run setup."
    exit 0
fi

# Create models directory if it doesn't exist
mkdir -p models

# Clone the repository if it doesn't exist
if [ ! -d "$MODEL_DIR" ]; then
    echo "Cloning Qwen3-0.6B repository..."
    git clone "$REPO_URL" "$MODEL_DIR"
    echo -e "${GREEN}✓ Repository cloned${NC}"
else
    echo -e "${YELLOW}Repository already exists at $MODEL_DIR${NC}"
fi

# Navigate to the model directory
cd "$MODEL_DIR"

# Run make concat to assemble model files
echo
echo "Assembling model files from parts (this may take a few minutes)..."
if [ ! -f "Makefile" ]; then
    echo "Error: Makefile not found in $MODEL_DIR"
    echo "Creating Makefile with concat target..."
    cat > Makefile << 'EOF'
concat:
	ls model.safetensors.part.* | sort -V | xargs cat > model.safetensors
EOF
fi

make concat
echo -e "${GREEN}✓ Model assembled successfully${NC}"

# Verify the model file was created
cd - > /dev/null
if [ -f "$MODEL_DIR/model.safetensors" ]; then
    MODEL_SIZE=$(du -h "$MODEL_DIR/model.safetensors" | cut -f1)
    echo
    echo -e "${GREEN}=== Setup Complete ===${NC}"
    echo "Model location: $MODEL_DIR/model.safetensors"
    echo "Model size: $MODEL_SIZE"
    echo
    echo "Next steps:"
    echo "  1. Run: uv run python scripts/verify_qwen3.py"
    echo "  2. Check the verification output"
else
    echo "Error: model.safetensors was not created"
    exit 1
fi
