# 🧠 NeurJournal

**Watch your thoughts light up the brain in real-time.**

NeurJournal maps journal entries to fMRI-predicted brain activations using Meta's [TRIBE v2](https://github.com/facebookresearch/TRIBE-2) deep multimodal brain encoder. Type a thought, and see which cortical regions activate on an interactive 3D brain — instantly.

> Built at Founders Inc Night Hacks. Powered by LLaMA 3.2-3B + Wav2Vec-BERT → 20,484 cortical vertices.

---

## What Makes This Different

Most "brain visualization" apps are decorative — they animate random regions with no scientific basis. NeurJournal uses a **research-grade brain encoder** (Meta TRIBE v2) that predicts actual fMRI vertex activations from text and audio. The same model used in neuroscience research papers, now in a journaling app.

**Every activation you see is real.** There are no fabricated values, no random noise, no keyword heuristics. If the TRIBE model isn't available, the app tells you honestly rather than faking results.

| Feature | NeurJournal | Typical "Brain App" |
|---------|------------|---------------------|
| Brain model | Meta TRIBE v2 (fMRI prediction) | Random/keyword mapping |
| Resolution | 20,484 cortical vertices | ~10 labeled blobs |
| Data integrity | 100% real model predictions, zero fabrication | Often decorative/random |
| Longitudinal tracking | Cognitive weather + trends over sessions | None |
| Scientific backing | fsaverage5 mesh, Destrieux atlas | Decorative |

---

## Key Features

### Brain Activation Analysis
Write a journal entry and click Analyze — TRIBE v2 predicts fMRI vertex activations from your text. The 3D brain visualizes real model output, not approximations.

### Cognitive Weather Dashboard
A longitudinal "weather report" for your brain — aggregates recent sessions to show dominant regions, rising/falling trends, and top emotions over time. Displayed at the top of the Journal tab.

### 3D Cortical Visualization
Interactive fsaverage5 mesh with inflated/pial views, orbit controls, region tooltips, connectivity lines between co-active regions, and an inferno-inspired colormap.

### Multi-Domain Platform
6 pre-built domain adapters — Therapy, Education, UX Research, Neuromarketing, Meditation, Sports — each with tailored prompts, region labels, and subject tracking.

### Semantic Memory System
Upload conversations, query them with natural language, and watch the AI-generated answer activate the brain. Uses sentence-transformers for semantic search with speaker-aware boosting.

---

## Architecture

```
┌──────────────────────────┐                 ┌──────────────────────────────┐
│   Browser (index.html)    │    REST API     │     FastAPI Backend           │
│                           │ ──────────────→ │                              │
│  Three.js 3D Brain        │  /analyze       │  ┌─────────────────────┐    │
│  20,484 vertex fsaverage5 │  /memory/query  │  │    TRIBE v2          │    │
│                           │  /cognitive-    │  │    LLaMA 3.2-3B      │    │
│  Cognitive Weather Card   │    weather      │  │    Wav2Vec-BERT       │    │
│                           │  /domains       │  │    (→ fMRI vertices)  │    │
│                           │  /subjects      │  └─────────────────────┘    │
│                           │  /upload        │           │                  │
│  All data from real       │ ←────────────── │  ┌───────▼──────────┐      │
│  TRIBE v2 predictions     │  activations +  │  │  Domain Adapters  │      │
│                           │  regions + LLM  │  │  (6 built-in)     │      │
│  Tabs: Journal · Memory   │  answer +       │  └──────────────────┘      │
│         Timeline · Patterns│  emotion        │  ┌──────────────────┐      │
│                           │                 │  │  SQLite + Sentence │      │
│  Voice Input · Dark/Light │                 │  │  Transformer       │      │
└──────────────────────────┘                 │  │  Embeddings         │      │
                                              │  └──────────────────┘      │
                                              └──────────────────────────────┘
```

---

## Quick Start

### 1. Setup

```bash
chmod +x setup.sh && ./setup.sh
```

### 2. Configure

```bash
# .env in project root
OPENAI_API_KEY=sk-...
```

### 3. Ingest Memories (optional)

```bash
source .venv/bin/activate
python backend/ingest_locomo.py
```

### 4. Run

```bash
source .venv/bin/activate
python -m backend.server
```

Open **http://localhost:8420** — the brain mesh loads in ~3 seconds. Write an entry and click Analyze to see real TRIBE v2 predictions.

> **Note**: The TRIBE model must be loaded on the backend for analysis to work. The app will clearly indicate when the model is unavailable rather than showing fabricated data.

---

## Domains

| Domain | Icon | Subject | Use Case |
|--------|------|---------|----------|
| Clinical Therapy | 🧠 | Patient | Track neural patterns during therapeutic conversations |
| Learning & Education | 📚 | Student | Monitor cognitive engagement during learning |
| UX Research | 🎨 | Participant | Measure neural responses to interfaces |
| Neuromarketing | 📊 | Consumer | Analyze responses to ads, brands, products |
| Meditation & Wellness | 🧘 | Practitioner | Track mindfulness states across practice |
| Sports Performance | ⚡ | Athlete | Analyze cognitive-motor patterns |

Add custom domains by extending `backend/domains.py`.

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status, model mode, available domains |
| `/analyze` | POST | Text → brain activation (TRIBE v2 only, 503 if unavailable) |
| `/memory/query` | POST | Semantic search + LLM answer + brain activation |
| `/cognitive-weather` | GET | Longitudinal brain weather report |
| `/domains` | GET | List domain adapters |
| `/subjects` | GET/POST | Manage subjects per domain |
| `/subjects/{id}/history` | GET | Activation history |
| `/subjects/{id}/trends` | GET | Longitudinal trends |
| `/sessions` | POST | Start new session |
| `/upload` | POST | Upload conversation files |
| `/emotion` | POST | Emotion classification |

---

## File Structure

```
neurojournal/
├── index.html              # Single-page app (Three.js, all UI)
├── brain_mesh.json         # fsaverage5 cortical mesh
├── setup.sh                # One-command setup
├── .env                    # API keys (not committed)
├── backend/
│   ├── server.py           # FastAPI server + TRIBE orchestration
│   ├── database.py         # SQLite persistence
│   ├── domains.py          # 6 domain adapters
│   ├── memory_store.py     # Semantic search (sentence-transformers)
│   ├── ingest_locomo.py    # LoCoMo dataset parser
│   └── requirements.txt    # Python dependencies
└── README.md
```

---

## Tech Stack

- **Brain Encoding**: Meta TRIBE v2 (LLaMA 3.2-3B + Wav2Vec-BERT 2.0)
- **Backend**: FastAPI + Uvicorn, SQLite
- **Frontend**: Vanilla JS, Three.js (3D), Web Speech API
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Emotion**: distilroberta (j-hartmann)
- **LLM**: GPT-4o-mini (text summaries only — never used to fabricate activations)

## License

TRIBE v2 model: CC BY-NC 4.0 (non-commercial research).
