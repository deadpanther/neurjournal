# NeuroJournal — Emotional Activation Mapping with Meta TRIBE v2

> Built for Founders Inc Night Hacks

**What it does:** Journal entries → TRIBE v2 brain encoding → cortical activation map → emotional pattern detection over time.

## Architecture

```
┌─────────────────┐     POST /analyze      ┌──────────────────────┐
│   React Frontend │ ──────────────────────→ │  FastAPI Backend      │
│   (neurojournal)  │ ←────────────────────── │                      │
│                   │    region activations   │  ┌────────────────┐  │
│  • Brain viz      │                         │  │  TRIBE v2       │  │
│  • Timeline       │                         │  │  (text→fMRI)   │  │
│  • Patterns       │                         │  │                │  │
└─────────────────┘                         │  │  LLaMA 3.2-3B  │  │
                                             │  │  + Brain Head   │  │
                                             │  └────────────────┘  │
                                             │         │            │
                                             │  ┌──────▼─────────┐  │
                                             │  │ Anthropic API   │  │
                                             │  │ (summary/label) │  │
                                             │  └────────────────┘  │
                                             └──────────────────────┘
```

## Quick Start (M4 Pro Mac)

### 1. Setup (run once, ~10 min)

```bash
chmod +x setup.sh
./setup.sh
```

This installs PyTorch, TRIBE v2, and all dependencies. First run downloads ~10GB of models.

**Prerequisites:**
- HuggingFace account + token (for LLaMA 3.2-3B access)
- Accept LLaMA license: https://huggingface.co/meta-llama/Llama-3.2-3B
- Anthropic API key (for emotion labeling)

### 2. Run Backend

```bash
source .venv/bin/activate
export ANTHROPIC_API_KEY=sk-ant-...
python backend/server.py
```

Server starts on `http://localhost:8420`. Check health: `curl localhost:8420/health`

### 3. Open Frontend

Open the `neurojournal.jsx` artifact in Claude.ai, or serve it locally.

The frontend auto-detects the backend:
- **Green "TRIBE v2 LIVE"** → Running real brain encoding
- **Yellow "API MODE"** → Backend running, TRIBE failed to load, using Anthropic
- **Purple "BROWSER"** → No backend, direct Anthropic API from browser

## How TRIBE v2 Integration Works

1. **Text input** → saved to temp `.txt` file
2. **gTTS** converts text to speech audio
3. **Whisper** transcribes audio back to word-level events with timestamps
4. **LLaMA 3.2-3B** extracts text features (contextual embeddings at layers 0.5, 0.75, 1.0)
5. **TRIBE v2 Transformer** maps features to fsaverage5 cortical mesh (~20,484 vertices)
6. **ROI mapping** aggregates vertex activations into 10 named brain regions
7. **Anthropic API** provides human-readable emotion labels and summary

## Files

```
neurojournal/
├── setup.sh              # One-command setup
├── backend/
│   ├── server.py          # FastAPI + TRIBE v2 inference
│   └── requirements.txt   # Python deps
├── neurojournal.jsx       # React frontend (Claude artifact)
└── README.md
```

## Demo Script (for pitch)

1. Show app with pre-seeded entries, click through brain activations
2. Type a new journal entry live → watch brain regions light up
3. Switch to Patterns tab → show detected emotional trends
4. Point out the green "TRIBE v2 LIVE" badge = real brain encoding
5. Pitch: "Every journal entry gets a neuroscience-grade activation fingerprint"

## What's Next

- [ ] Supermemory/Mem0 integration for persistent cross-session memory
- [ ] Mood-congruent memory retrieval (find past entries by neural similarity)
- [ ] Therapeutic content recommendation based on activation profiles
- [ ] Audio/video journaling (TRIBE v2 supports all three modalities)

## License

TRIBE v2 is CC BY-NC 4.0 (non-commercial research only).
