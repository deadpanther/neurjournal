# NeurJournal — Real-Time Brain Activation Mapping

> Built at Founders Inc Night Hacks

**NeurJournal** maps text to real brain activations using Meta's **TRIBE v2** deep multimodal brain encoding model. A therapist asks questions about a patient's memories, and the app retrieves relevant memories, generates answers, and visualizes predicted fMRI cortical activity on an interactive 3D brain — in real time.

## Architecture

```
┌──────────────────────┐                    ┌──────────────────────────┐
│   Browser (index.html)│   POST /analyze    │    FastAPI Backend        │
│                       │ ──────────────────→│                          │
│  Three.js 3D Brain    │   POST /memory/    │  ┌──────────────────┐   │
│  fsaverage5 20k verts │    query           │  │   TRIBE v2        │   │
│                       │ ←──────────────────│  │   Wav2Vec-BERT    │   │
│  Tabs:                │  vertex activations│  │   (audio → fMRI)  │   │
│  • Journal            │  + regions + answer│  └──────────────────┘   │
│  • Memory             │                    │           │              │
│  • Timeline           │                    │  ┌────────▼─────────┐   │
│  • Patterns           │                    │  │  OpenAI API       │   │
│                       │                    │  │  Whisper + GPT-4o │   │
└──────────────────────┘                    │  └──────────────────┘   │
                                             │           │              │
                                             │  ┌────────▼─────────┐   │
                                             │  │  Memory Store     │   │
                                             │  │  TF-IDF + 407     │   │
                                             │  │  LoCoMo memories  │   │
                                             │  └──────────────────┘   │
                                             └──────────────────────────┘
```

## How It Works

1. **Memory ingestion** — 2 conversations from the LoCoMo dataset (Caroline & Melanie, Jon & Gina) are parsed into 407 searchable memory records
2. **Therapist query** — e.g. *"Jon, how are you managing stress these days?"*
3. **Memory retrieval** — TF-IDF search with speaker-aware boosting returns the top 5 relevant memories
4. **Answer generation** — GPT-4o-mini synthesizes a grounded answer from the retrieved memories
5. **Brain encoding** — TRIBE v2 converts the answer to speech (gTTS), transcribes it (OpenAI Whisper), extracts audio features (Wav2Vec-BERT), and predicts fMRI activity across 20,484 cortical vertices
6. **Visualization** — Activations are painted onto an interactive 3D brain (Three.js + fsaverage5 mesh) with an inferno colormap
7. **Timeline** — Each query's activations are stored, enabling progression tracking with sparkline charts, smooth brain animation playback, and delta badges showing which regions changed

## Features

- **3D Brain Visualization** — Interactive fsaverage5 cortical mesh with orbit controls, inflated/pial views, auto-rotation
- **Memory Tab** — Query patient memories, view retrieved evidence, get AI-generated answers with TRIBE brain activations
- **Query History** — Click any past query to instantly restore its brain state
- **Timeline Tab** — Sparkline chart showing activation progression across queries, with playback controls to animate the brain through the session
- **Delta Badges** — When stepping through the timeline, floating indicators show which regions increased or decreased
- **Follow-up Suggestions** — Context-aware follow-up questions appear after each query to guide the therapist
- **Patterns Tab** — Detects neural coupling, rising activations, and baseline patterns across journal entries
- **Journal Tab** — Free-text emotional journaling with real-time TRIBE activation mapping

## Quick Start

### 1. Setup

```bash
chmod +x setup.sh
./setup.sh
```

Downloads and installs TRIBE v2, PyTorch, and all dependencies (~10 min first run).

**Prerequisites:**
- Python 3.10+
- HuggingFace account + token (`huggingface-cli login`)
- OpenAI API key (for Whisper transcription + GPT-4o-mini)

### 2. Configure

```bash
# Create .env in project root
OPENAI_API_KEY=sk-...
```

### 3. Ingest Memories

```bash
source .venv/bin/activate
python backend/ingest_locomo.py
```

Downloads the LoCoMo dataset and generates `backend/memories.json` with 407 memories.

### 4. Run

```bash
source .venv/bin/activate
python backend/server.py
```

Open `http://localhost:8420` in your browser.

Status indicators:
- **Green "TRIBE v2 LIVE"** — Real brain encoding active
- **Yellow "API MODE"** — Backend running, TRIBE not loaded, using LLM fallback
- **Purple "TRIBE v2"** — No backend, demo mode with simulated activations

## Files

```
neurojournal/
├── index.html                  # Full frontend (Three.js + UI)
├── brain_mesh.json             # fsaverage5 cortical mesh data
├── setup.sh                    # One-command environment setup
├── .env                        # API keys (not committed)
├── backend/
│   ├── server.py               # FastAPI server + TRIBE v2 inference
│   ├── memory_store.py         # TF-IDF memory search with speaker boosting
│   ├── ingest_locomo.py        # LoCoMo dataset parser
│   └── memories.json           # 407 ingested memory records
└── README.md
```

## TRIBE v2 Pipeline

```
Text input
  → gTTS (text-to-speech)
  → OpenAI Whisper (word-level timestamps → .tsv cache)
  → Wav2Vec-BERT (audio embeddings on CPU)
  → TRIBE v2 Transformer (multimodal → cortical prediction)
  → 20,484 vertex activations on fsaverage5
  → 10 ROI aggregations (Prefrontal, Amygdala, Hippocampus, etc.)
```

## Demo Script

1. Open the app — show the 3D brain with pre-seeded journal entries
2. Switch to **Memory** tab — click a sample question like *"Caroline, what career path are you pursuing?"*
3. Watch the brain light up with real TRIBE activations while memories are retrieved
4. Click a follow-up suggestion to continue the conversation
5. Switch to **Timeline** tab — hit Play to animate the brain morphing between emotional states
6. Point out delta badges: *"See how Hippocampus drops 22% when we shift from memory recall to career planning"*
7. Pitch: *"Every therapist question gets a neuroscience-grade brain activation map — showing exactly which neural circuits fire during recall"*

## Use Cases

- **Therapy tools** — Visualize which brain regions activate when patients discuss different topics
- **Neuroscience education** — Interactive brain encoding demonstrations
- **Memory research** — Study how different types of recall activate distinct neural circuits
- **Emotional profiling** — Track activation patterns over time to detect emotional trends

## License

TRIBE v2 model is CC BY-NC 4.0 (non-commercial research only).
