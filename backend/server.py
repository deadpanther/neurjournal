"""
NeuroJournal Backend — TRIBE v2 Text Inference Server

Runs TRIBE v2 locally for text-only brain activation prediction.
Falls back to Anthropic API emotion analysis if TRIBE can't load.

Usage:
    pip install fastapi uvicorn
    # Plus tribev2 deps (see setup.sh)
    python server.py
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Brain Region ROI Mapping ───
# Maps fsaverage5 vertex ranges to named brain regions.
# Based on Destrieux atlas parcellation of the ~20k vertex fsaverage5 mesh.
# Each region has associated emotional/cognitive functions from neuroscience literature.
BRAIN_REGIONS = {
    "prefrontal_cortex": {
        "label": "Prefrontal Cortex",
        "vertex_ranges": [(0, 1200)],  # approximate dorsolateral/ventromedial PFC vertices
        "functions": ["reasoning", "planning", "calm", "decision_making"],
        "x": 28, "y": 28,
    },
    "anterior_cingulate": {
        "label": "Anterior Cingulate Cortex",
        "vertex_ranges": [(1200, 1800)],
        "functions": ["conflict", "motivation", "sadness", "error_detection"],
        "x": 40, "y": 35,
    },
    "insula": {
        "label": "Insula",
        "vertex_ranges": [(1800, 2400)],
        "functions": ["disgust", "empathy", "self_awareness", "interoception"],
        "x": 58, "y": 45,
    },
    "temporal_lobe": {
        "label": "Temporal Lobe",
        "vertex_ranges": [(2400, 4200)],
        "functions": ["language", "comprehension", "social", "semantics"],
        "x": 72, "y": 50,
    },
    "amygdala": {
        "label": "Amygdala",
        "vertex_ranges": [(4200, 4800)],
        "functions": ["fear", "anxiety", "anger", "threat_detection"],
        "x": 50, "y": 62,
    },
    "hippocampus": {
        "label": "Hippocampus",
        "vertex_ranges": [(4800, 5400)],
        "functions": ["memory", "nostalgia", "learning", "spatial_memory"],
        "x": 45, "y": 68,
    },
    "nucleus_accumbens": {
        "label": "Nucleus Accumbens",
        "vertex_ranges": [(5400, 5800)],
        "functions": ["joy", "reward", "excitement", "motivation"],
        "x": 48, "y": 52,
    },
    "parietal_lobe": {
        "label": "Parietal Lobe",
        "vertex_ranges": [(5800, 7800)],
        "functions": ["attention", "spatial", "integration", "body_awareness"],
        "x": 55, "y": 22,
    },
    "occipital_lobe": {
        "label": "Visual Cortex",
        "vertex_ranges": [(7800, 9600)],
        "functions": ["imagery", "visualization", "creativity", "visual_processing"],
        "x": 70, "y": 30,
    },
    "motor_cortex": {
        "label": "Motor Cortex",
        "vertex_ranges": [(9600, 10484)],
        "functions": ["action", "restlessness", "energy", "motor_planning"],
        "x": 38, "y": 18,
    },
}

# ─── Global Model State ───
tribe_model = None
model_mode = "loading"  # "tribe", "fallback", "loading"

# ─── Memory Store ───
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_store import MemoryStore

memory_store = MemoryStore()


def load_tribe_model():
    """Attempt to load TRIBE v2 from HuggingFace."""
    global tribe_model, model_mode
    try:
        logger.info("Loading TRIBE v2 model from HuggingFace...")
        from tribev2 import TribeModel

        # Skip text features (LLaMA 3.2-3B) if user lacks HF access.
        # Audio features via Wav2Vec-BERT are ungated and still produce
        # meaningful cortical activation maps.
        # Use audio-only features to avoid downloading LLaMA (~6GB).
        # Wav2Vec-BERT is already cached and produces real cortical activations.
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config_update = {
            "data.features_to_use": ["audio"],
            "data.audio_feature.device": device,
            "data.text_feature.device": device,
            "data.num_workers": 2,
            "data.batch_size": 1,
        }
        logger.info(f"Running TRIBE with audio features only (Wav2Vec-BERT) on {device}")

        model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder="./cache",
            device="auto",
            config_update=config_update,
        )
        tribe_model = model
        model_mode = "tribe"
        logger.info("TRIBE v2 loaded successfully! Running in TRIBE mode.")
        return True
    except Exception as e:
        logger.warning(f"Could not load TRIBE v2: {e}")
        logger.info("Falling back to Anthropic API mode.")
        model_mode = "fallback"
        return False


def _transcribe_with_openai_whisper(audio_path: Path) -> None:
    """Call OpenAI Whisper API and write a .tsv next to the audio file.

    TRIBE's ExtractWordsFromAudio checks for a .tsv cache file before
    launching local WhisperX.  By writing the transcript ourselves, TRIBE
    skips the slow local path entirely.
    """
    import httpx
    import pandas as pd

    tsv_path = audio_path.with_suffix(".tsv")
    if tsv_path.exists():
        return

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set for Whisper transcription")

    with open(audio_path, "rb") as f:
        resp = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "model": "whisper-1",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
                "language": "en",
            },
            files={"file": (audio_path.name, f, "audio/mpeg")},
            timeout=30.0,
        )
    result = resp.json()

    words = []
    sentence = result.get("text", "").replace('"', "")
    for w in result.get("words", []):
        if "start" not in w:
            continue
        words.append({
            "text": w["word"].replace('"', ""),
            "start": w["start"],
            "duration": w["end"] - w["start"],
            "sequence_id": 0,
            "sentence": sentence,
        })

    df = pd.DataFrame(words)
    df.to_csv(tsv_path, sep="\t", index=False)
    logger.info(f"OpenAI Whisper transcript written to {tsv_path}")


def _predict_with_tribe_sync(text: str) -> dict:
    """Run text through TRIBE v2, using OpenAI Whisper for transcription."""
    import tempfile
    import glob as globmod
    import langdetect
    from pathlib import Path as P

    _orig_detect = langdetect.detect
    langdetect.detect = lambda *a, **kw: "en"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        text_path = f.name

    try:
        # Step 1: Let TRIBE do gTTS (fast) — this creates the audio.mp3
        # We call get_events_dataframe but it will stall at WhisperX.
        # Instead, we replicate just the gTTS part, then pre-cache the transcript.
        from tribev2.demo_utils import TextToEvents
        from exca import TaskInfra
        cache_folder = getattr(tribe_model, 'cache_folder', './cache')
        tts = TextToEvents(
            text=P(text_path).read_text(),
            infra={"folder": cache_folder, "mode": "retry"},
        )
        # Trigger gTTS only — the infra computes a UID folder for caching
        uid_folder = P(tts.infra.uid_folder(create=True))
        audio_path = uid_folder / "audio.mp3"
        if not audio_path.exists():
            from gtts import gTTS
            tts_audio = gTTS(text, lang="en")
            tts_audio.save(str(audio_path))
            logger.info(f"Wrote TTS audio to {audio_path}")

        # Step 2: Pre-cache transcript via OpenAI Whisper API
        _transcribe_with_openai_whisper(audio_path)

        # Step 3: Now let TRIBE run — it finds the .tsv and skips WhisperX
        events = tribe_model.get_events_dataframe(text_path=text_path)
        preds, segments = tribe_model.predict(events, verbose=False)
        avg_activation = preds.mean(axis=0)
        region_activations = compute_region_activations(avg_activation)

        norm = avg_activation - avg_activation.min()
        denom = norm.max()
        if denom > 0:
            norm = norm / denom
        vertex_activations_list = np.round(norm, 4).tolist()

        return {
            "mode": "tribe",
            "regions": region_activations,
            "vertex_activations": vertex_activations_list,
            "n_vertices": int(preds.shape[1]),
            "n_timesteps": int(preds.shape[0]),
            "raw_stats": {
                "mean": float(avg_activation.mean()),
                "std": float(avg_activation.std()),
                "max": float(avg_activation.max()),
                "min": float(avg_activation.min()),
            },
        }
    finally:
        langdetect.detect = _orig_detect
        os.unlink(text_path)


def predict_with_tribe(text: str) -> dict:
    """Run TRIBE v2 — no timeout, let it finish on CPU."""
    return _predict_with_tribe_sync(text)


def compute_region_activations(vertex_activations: np.ndarray) -> dict:
    """Map vertex-level activations to named brain regions."""
    n_vertices = len(vertex_activations)
    regions = {}

    for region_key, region_info in BRAIN_REGIONS.items():
        # Collect activations for this region's vertices
        region_vals = []
        for start, end in region_info["vertex_ranges"]:
            # Scale vertex ranges proportionally if mesh size differs
            scaled_start = int(start * n_vertices / 10484)
            scaled_end = int(end * n_vertices / 10484)
            scaled_end = min(scaled_end, n_vertices)
            if scaled_start < n_vertices:
                region_vals.extend(vertex_activations[scaled_start:scaled_end].tolist())

        if region_vals:
            arr = np.array(region_vals)
            # Normalize activation to 0-1 range using z-score then sigmoid
            mean_act = float(arr.mean())
            # Use percentile rank across all vertices as activation level
            activation = float(
                np.mean(vertex_activations < mean_act)
            )  # percentile rank
        else:
            activation = 0.0

        regions[region_key] = {
            "label": region_info["label"],
            "activation": round(activation, 4),
            "functions": region_info["functions"],
            "position": {"x": region_info["x"], "y": region_info["y"]},
            "mean_raw": round(mean_act, 6) if region_vals else 0.0,
        }

    return regions


async def llm_chat(prompt: str, max_tokens: int = 1000) -> str:
    """Call the best available LLM: OpenAI first, then Anthropic."""
    import httpx

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if openai_key:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                json={"model": "gpt-4o-mini", "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=30.0,
            )
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    if anthropic_key:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json", "x-api-key": anthropic_key,
                          "anthropic-version": "2023-06-01"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=30.0,
            )
        data = resp.json()
        return data["content"][0]["text"]

    raise HTTPException(status_code=500, detail="No LLM API key set (OPENAI_API_KEY or ANTHROPIC_API_KEY)")


async def predict_with_fallback(text: str) -> dict:
    """Use LLM API as fallback when TRIBE isn't available."""
    all_functions = set()
    for r in BRAIN_REGIONS.values():
        all_functions.update(r["functions"])

    prompt = f"""Analyze this journal entry for emotional and cognitive activations.
Return ONLY a JSON object mapping these exact function names to activation values (0.0-1.0):
{sorted(all_functions)}

Only include functions with activation > 0.1. Also include:
- "summary": a <15 word summary
- "dominant": the single strongest function name

Journal entry: "{text}"

Return ONLY valid JSON, no markdown, no explanation."""

    raw = await llm_chat(prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)

    summary = parsed.pop("summary", "Journal entry")
    dominant = parsed.pop("dominant", "calm")

    regions = {}
    for region_key, region_info in BRAIN_REGIONS.items():
        max_activation = 0.0
        for func in region_info["functions"]:
            if func in parsed and parsed[func] > max_activation:
                max_activation = parsed[func]

        regions[region_key] = {
            "label": region_info["label"],
            "activation": round(max_activation, 4),
            "functions": region_info["functions"],
            "position": {"x": region_info["x"], "y": region_info["y"]},
        }

    return {
        "mode": "fallback",
        "regions": regions,
        "summary": summary,
        "dominant": dominant,
        "raw_emotions": parsed,
    }


# ─── FastAPI App ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: try to load TRIBE
    load_tribe_model()
    yield


app = FastAPI(title="NeuroJournal API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    mode: str  # "tribe" or "fallback"
    regions: dict
    vertex_activations: list[float] | None = None
    summary: str | None = None
    dominant: str | None = None
    raw_emotions: dict | None = None
    n_vertices: int | None = None
    n_timesteps: int | None = None
    raw_stats: dict | None = None
    processing_time_ms: float = 0


@app.get("/health")
async def health():
    return {"status": "ok", "model_mode": model_mode}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start = time.time()

    if model_mode == "tribe" and tribe_model is not None:
        try:
            result = predict_with_tribe(req.text)
            # Use LLM for summary/dominant since TRIBE only gives activations
            fallback_result = await predict_with_fallback(req.text)
            result["summary"] = fallback_result.get("summary")
            result["dominant"] = fallback_result.get("dominant")
            result["raw_emotions"] = fallback_result.get("raw_emotions")
        except Exception as e:
            logger.error(f"TRIBE prediction failed: {e}")
            result = await predict_with_fallback(req.text)
    else:
        result = await predict_with_fallback(req.text)

    elapsed = (time.time() - start) * 1000
    return AnalyzeResponse(**result, processing_time_ms=round(elapsed, 1))


@app.get("/regions")
async def get_regions():
    """Return the brain region mapping for the frontend."""
    return BRAIN_REGIONS


# ─── Memory Endpoints ───


class MemoryQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    run_tribe: bool = True


@app.post("/memory/query")
async def memory_query(req: MemoryQueryRequest):
    if not memory_store.ready:
        raise HTTPException(status_code=503, detail="Memory store not loaded. Run ingest_locomo.py first.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start = time.time()

    retrieved = memory_store.search(req.query, top_k=req.top_k)
    memory_context = "\n".join(
        f"- [{m['speaker']}, {m['date']}] {m['text']}" for m in retrieved
    )
    speakers = set()
    for m in retrieved:
        speakers.update(m.get("speakers", []))

    # Generate answer from memories using LLM
    generated_answer = ""
    prompt = f"""You are a memory-augmented assistant. Given retrieved memories from long-term conversations between {', '.join(speakers)}, answer the user's question concisely and accurately.

Retrieved memories:
{memory_context}

Question: {req.query}

Answer in 1-3 sentences based ONLY on the memories above. If the memories don't contain enough information, say so."""

    try:
        generated_answer = await llm_chat(prompt, max_tokens=300)
    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}")
        generated_answer = f"Based on memories: {retrieved[0]['text']}" if retrieved else "No relevant memories found."

    # Run the generated answer through TRIBE (or fallback) for brain activations
    tribe_result = None
    if req.run_tribe and generated_answer:
        if model_mode == "tribe" and tribe_model is not None:
            try:
                tribe_result = predict_with_tribe(generated_answer)
            except Exception as e:
                logger.error(f"TRIBE on memory answer failed: {e}")
        if tribe_result is None:
            try:
                tribe_result = await predict_with_fallback(generated_answer)
            except Exception as e:
                logger.error(f"Fallback on memory answer failed: {e}")

    elapsed = (time.time() - start) * 1000
    return {
        "query": req.query,
        "retrieved_memories": retrieved,
        "generated_answer": generated_answer,
        "tribe": tribe_result,
        "processing_time_ms": round(elapsed, 1),
    }


@app.get("/memory/conversations")
async def memory_conversations():
    return {
        "ready": memory_store.ready,
        "conversations": memory_store.conversations,
        "total_memories": len(memory_store.memories),
    }


@app.get("/memory/all")
async def memory_all():
    return {"memories": memory_store.memories}


@app.get("/memory/sample-questions")
async def memory_sample_questions():
    return {"questions": memory_store.sample_questions}


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@app.get("/brain_mesh.json")
async def get_brain_mesh():
    mesh_path = PROJECT_ROOT / "brain_mesh.json"
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="brain_mesh.json not found — run mesh export first")
    return FileResponse(mesh_path, media_type="application/json")


@app.get("/")
async def serve_index():
    return FileResponse(PROJECT_ROOT / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8420, log_level="info")
