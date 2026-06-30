#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Digital Twin Consumer — Quick Start Script
# Run this in Git Bash: bash quickstart.sh
# ═══════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║   Digital Twin Consumer Behaviour Pipeline            ║"
echo "║   Karina Dhaka | AI in Consumer Behaviour            ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Check conda
if false && ! command -v conda &>/dev/null; then
    echo "❌ Conda not found. Open Anaconda Prompt instead."
    exit 1
fi

# Activate environment (create if doesn't exist)
ENV_NAME="digital_twin"
if conda env list | grep -q "$ENV_NAME"; then
    echo "✓ Environment '$ENV_NAME' found"
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -n $ENV_NAME python=3.10 -y
fi

source activate $ENV_NAME 2>/dev/null || conda activate $ENV_NAME

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt -q
echo "✓ Requirements installed"

# Check Ollama
echo ""
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✓ Ollama is running"
    echo "  Available models:"
    curl -s http://localhost:11434/api/tags | python -c "
import json,sys
data=json.load(sys.stdin)
for m in data.get('models',[]):
    print('   -', m['name'])
"
else
    echo "⚠️  Ollama not running!"
    echo "   Open a new Git Bash window and run: ollama serve"
    echo "   Then rerun this script."
    echo ""
    echo "   If you don't have models yet:"
    echo "   ollama pull llama3"
    echo "   ollama pull phi3"
fi

# Run pipeline
echo ""
echo "═══ RUNNING PIPELINE ═══"
echo ""

echo "[Step 0] Downloading datasets..."
python pipeline/00_download_data.py
echo ""

echo "[Step 1] Extracting persona vectors..."
python pipeline/01_persona_extraction.py
echo ""

echo "[Step 2] Building RAG index..."
python pipeline/02_build_rag_index.py
echo ""

echo "[Step 3] Running model experiments..."
echo "         (This takes 20-60 min. Progress shown below.)"
python pipeline/03_run_experiments.py
echo ""

echo "[Step 4] Computing alignment metrics..."
python metrics/04_alignment_metrics.py
echo ""

echo "╔═══════════════════════════════════════════════════════╗"
echo "║   ✓ PIPELINE COMPLETE!                                ║"
echo "║                                                       ║"
echo "║   Results in: outputs/                               ║"
echo "║   - model_ladder_plot.png     ← your main figure     ║"
echo "║   - nudge_gap_plot.png        ← H3 evidence          ║"
echo "║   - hypothesis_summary.txt    ← H1-H5 results        ║"
echo "║   - alignment_scores.csv      ← full data            ║"
echo "║                                                       ║"
echo "║   Open notebook: jupyter notebook                     ║"
echo "║   File: notebooks/05_results_analysis.ipynb           ║"
echo "╚═══════════════════════════════════════════════════════╝"
