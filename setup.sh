#!/bin/bash
# NeuroJournal — Hackathon Setup Script
# Run this on your M4 Pro Mac to get everything working

set -e

echo "╔══════════════════════════════════════════╗"
echo "║   NeuroJournal — TRIBE v2 Setup          ║"
echo "║   Founders Inc Night Hacks               ║"
echo "╚══════════════════════════════════════════╝"

# ─── 1. Python Environment ───
echo ""
echo "→ Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# ─── 2. Core Dependencies ───
echo "→ Installing core dependencies..."
pip install --upgrade pip
pip install fastapi uvicorn httpx pydantic

# ─── 3. PyTorch (MPS-enabled for Apple Silicon) ───
echo "→ Installing PyTorch with MPS support..."
pip install torch torchvision

# ─── 4. TRIBE v2 ───
# Clone into facebook-tribev2 (not "tribev2"): a folder named tribev2/ here becomes a
# PEP 420 *namespace* package on sys.path and shadows the real tribev2 package when
# you run Python from the project root (e.g. setup verification below).
echo "→ Cloning and installing TRIBE v2..."
TRIBE_SRC="facebook-tribev2"
if [ ! -d "$TRIBE_SRC" ]; then
    git clone https://github.com/facebookresearch/tribev2.git "$TRIBE_SRC"
fi
cd "$TRIBE_SRC"
pip install -e .
cd ..

# ─── 5. HuggingFace Auth (needed for LLaMA 3.2-3B) ───
echo ""
echo "→ HuggingFace login required for LLaMA 3.2-3B access"
echo "  You need a READ token from: https://huggingface.co/settings/tokens"
echo "  You also need to accept the LLaMA license at:"
echo "  https://huggingface.co/meta-llama/Llama-3.2-3B"
echo ""
pip install huggingface_hub
huggingface-cli login

# ─── 6. spaCy model (needed for word extraction) ───
echo "→ Installing spaCy English model..."
pip install spacy
python3 -m spacy download en_core_web_sm

# ─── 7. Verify ───
echo ""
echo "→ Verifying installation..."
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CUDA available: {torch.cuda.is_available()}')
from tribev2 import TribeModel
print('TRIBE v2 imported successfully!')
print()
print('Setup complete! Run the server with:')
print('  export ANTHROPIC_API_KEY=your_key_here')
print('  python backend/server.py')
"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Setup Complete!                        ║"
echo "║                                          ║"
echo "║   1. export ANTHROPIC_API_KEY=sk-...     ║"
echo "║   2. python backend/server.py            ║"
echo "║   3. Open the React app                  ║"
echo "║                                          ║"
echo "║   First run will download ~10GB of       ║"
echo "║   models (LLaMA 3.2-3B + TRIBE v2).     ║"
echo "║   Subsequent runs use cache.             ║"
echo "╚══════════════════════════════════════════╝"
