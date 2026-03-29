# NeurJournal — Multi-Domain Brain Activation Platform

> Built at Founders Inc Night Hacks

**NeurJournal** is an extensible platform that maps text to real brain activations using Meta's **TRIBE v2** deep multimodal brain encoding model. Switch between domains — therapy, education, UX research, neuromarketing, meditation, sports — each with tailored prompts, region labeling, and session tracking. Upload your own conversations, create subjects, and track longitudinal neural trends over time.

## Architecture

```
┌──────────────────────┐                    ┌──────────────────────────┐
│   Browser (index.html)│   REST API         │    FastAPI Platform       │
│                       │ ──────────────────→│                          │
│  Three.js 3D Brain    │   /analyze         │  ┌──────────────────┐   │
│  fsaverage5 20k verts │   /memory/query    │  │   TRIBE v2        │   │
│                       │   /domains         │  │   LLaMA 3.2-3B    │   │
│  Domain Switcher      │   /subjects        │  │   Wav2Vec-BERT    │   │
│  Subject Manager      │   /sessions        │  │   (→ fMRI)        │   │
│  File Upload          │   /activations     │  └──────────────────┘   │
│                       │   /upload          │           │              │
│  Tabs:                │ ←──────────────────│  ┌────────▼─────────┐   │
│  • Journal            │  activations +     │  │  Domain Adapters  │   │
│  • Memory             │  regions + answer  │  │  (6 built-in)     │   │
│  • Timeline           │  + emotion + trends│  └──────────────────┘   │
│  • Patterns           │                    │           │              │
│                       │                    │  ┌────────▼─────────┐   │
└──────────────────────┘                    │  │  SQLite + Memory  │   │
                                             │  │  Persistent store │   │
                                             │  │  + Embeddings     │   │
                                             │  └──────────────────┘   │
                                             └──────────────────────────┘
```

## Domains

NeurJournal ships with 6 pre-built domain adapters. Each customizes prompts, region labels, subject terminology, and sample questions:

| Domain | Icon | Subject Label | Use Case |
|--------|------|---------------|----------|
| Clinical Therapy | 🧠 | Patient | Track neural patterns during therapeutic conversations |
| Learning & Education | 📚 | Student | Monitor cognitive engagement during learning |
| UX Research | 🎨 | Participant | Measure neural responses to interfaces |
| Neuromarketing | 📊 | Consumer | Analyze responses to ads, brands, products |
| Meditation & Wellness | 🧘 | Practitioner | Track mindfulness states across practice |
| Sports Performance | ⚡ | Athlete | Analyze cognitive-motor patterns |

Custom domains can be added by extending `backend/domains.py`.

## How It Works

1. **Choose a domain** — Click a domain pill (Therapy, Education, UX, etc.) to configure the platform
2. **Create or select a subject** — Add patients, students, participants, etc.
3. **Memory ingestion** — Upload conversations (.json, .txt, .csv) or use pre-loaded LoCoMo data
4. **Query** — Ask a question in the context of your domain
5. **Memory retrieval** — Semantic search (sentence-transformers) with speaker-aware boosting
6. **Answer generation** — GPT-4o-mini synthesizes an answer using the domain's system prompt
7. **Brain encoding** — TRIBE v2 (LLaMA 3.2-3B + Wav2Vec-BERT) predicts fMRI across 20,484 cortical vertices
8. **Visualization** — Activations painted on interactive 3D brain with inferno colormap
9. **Persistence** — Every query, activation, and emotion is saved to SQLite for longitudinal tracking
10. **Trends** — View how a subject's brain patterns change over weeks of sessions

## Features

### Core
- **3D Brain Visualization** — Interactive fsaverage5 cortical mesh, inflated/pial views, orbit controls
- **Multi-Domain Platform** — 6 built-in domains, each with tailored prompts and region context
- **Subject Management** — Create and track subjects across sessions
- **Persistent Storage** — SQLite database for sessions, activations, and longitudinal data
- **File Upload** — Ingest custom conversations from JSON, TXT, or CSV files

### Analysis
- **Semantic Memory Search** — sentence-transformers embeddings with speaker-aware boosting
- **Emotion Detection** — Per-query emotion classification (distilroberta)
- **Brain Connectivity** — Curved lines between co-active regions
- **Activation Similarity** — Find queries with similar brain patterns (cosine similarity)
- **Longitudinal Trends** — Track how regions change over time for each subject
- **Multi-Query Comparison** — Side-by-side activation tables

### Interface
- **Memory Tab** — Query memories, view evidence, get AI answers with brain activations
- **Timeline Tab** — Sparkline chart with playback, delta badges, smooth brain animation
- **Follow-up Suggestions** — Context-aware next questions
- **Session Summary** — LLM-generated clinical summaries
- **Voice Input** — Web Speech API for hands-free querying
- **Region Tooltips** — Hover over 3D brain for region details
- **Dark/Light Theme** — Toggle between themes
- **Keyboard Shortcuts** — 1-4 tabs, Space play/pause, arrows navigate
- **Export Reports** — Download standalone HTML session reports

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status, model mode, available domains |
| `/domains` | GET | List all domain adapters |
| `/domains/{id}` | GET | Full domain config (prompts, labels, samples) |
| `/subjects` | GET/POST | List or create subjects (filter by `domain_id`) |
| `/subjects/{id}/history` | GET | Activation history for a subject |
| `/subjects/{id}/trends` | GET | Longitudinal region trends and emotion counts |
| `/sessions` | POST | Start a new session for a subject |
| `/activations` | POST | Save an activation record |
| `/analyze` | POST | Direct text → brain activation (TRIBE or fallback) |
| `/memory/query` | POST | Semantic memory search + answer + TRIBE activation |
| `/upload` | POST | Upload conversation files for memory ingestion |
| `/emotion` | POST | Classify text emotion |
| `/llm_chat` | POST | General LLM chat endpoint |

## Quick Start

### 1. Setup

```bash
chmod +x setup.sh
./setup.sh
```

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

### 4. Run

```bash
source .venv/bin/activate
python backend/server.py
```

Open `http://localhost:8420`. Select a domain, create a subject, and start querying.

## Files

```
neurojournal/
├── index.html                  # Full frontend (Three.js + domain UI)
├── brain_mesh.json             # fsaverage5 cortical mesh data
├── setup.sh                    # One-command environment setup
├── .env                        # API keys (not committed)
├── backend/
│   ├── server.py               # FastAPI platform server
│   ├── database.py             # SQLite persistence layer
│   ├── domains.py              # Domain adapter registry (6 domains)
│   ├── memory_store.py         # Semantic search (sentence-transformers)
│   ├── ingest_locomo.py        # LoCoMo dataset parser
│   ├── memories.json           # Ingested memory records
│   ├── neurjournal.db          # SQLite database (auto-created)
│   └── requirements.txt        # Python dependencies
└── README.md
```

## Extending to New Domains

Add a new domain in `backend/domains.py`:

```python
DOMAIN_REGISTRY["my_domain"] = {
    "id": "my_domain",
    "name": "My Custom Domain",
    "icon": "🔬",
    "description": "Description for your use case.",
    "subject_label": "Participant",
    "session_label": "Session",
    "query_placeholder": "Ask about...",
    "system_prompt": "You are an assistant for...",
    "region_labels": {
        "prefrontal": {"label": "Prefrontal Cortex", "context": "Your context here"},
        # ... more regions
    },
    "sample_questions": ["Example question 1?", "Example question 2?"],
}
```

The domain automatically appears in the frontend switcher on next server restart.

## License

TRIBE v2 model is CC BY-NC 4.0 (non-commercial research only).
