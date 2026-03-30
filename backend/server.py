"""
NeurJournal Backend — Multi-domain brain activation platform.

Extensible to therapy, education, UX research, neuromarketing, meditation, sports.
TRIBE v2 for fMRI prediction. Persistent sessions via SQLite. Domain adapter system.

Usage:
    pip install fastapi uvicorn
    python server.py
"""

import json
import logging
import os
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

# Suppress noisy third-party warnings that clutter the terminal
warnings.filterwarnings("ignore", category=UserWarning, module="neuralset")
warnings.filterwarnings("ignore", category=FutureWarning, module="x_transformers")
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*LabelEncoder.*")
warnings.filterwarnings("ignore", message=".*Missing events.*")
warnings.filterwarnings("ignore", message=".*LOAD REPORT.*")
warnings.filterwarnings("ignore", message=".*event_types has not been set.*")

# Also suppress via logging for warnings that go through the logging system
for _noisy in ("neuralset.extractors.base",):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from backend.database import SessionDB
    from backend.domains import DOMAIN_REGISTRY, get_domain, list_domains as list_domain_configs, get_system_prompt
except ImportError:
    from database import SessionDB
    from domains import DOMAIN_REGISTRY, get_domain, list_domains as list_domain_configs, get_system_prompt

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
model_mode = "loading"  # "tribe", "unavailable", "loading"
emotion_classifier = None

# ─── Persistent Storage + Domains ───
db = SessionDB()

# ─── Memory Store ───
try:
    from backend.memory_store import MemoryStore
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from memory_store import MemoryStore

memory_store = MemoryStore()


def load_emotion_model():
    """Load lightweight emotion classifier (distilroberta, ~300MB)."""
    global emotion_classifier
    try:
        from transformers import pipeline
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1,
        )
        logger.info("Emotion classifier loaded (distilroberta)")
    except Exception as e:
        logger.warning(f"Could not load emotion classifier: {e}")


def load_tribe_model():
    """Attempt to load TRIBE v2 from HuggingFace with full features (audio + text)."""
    global tribe_model, model_mode
    try:
        logger.info("Loading TRIBE v2 model from HuggingFace...")
        from tribev2 import TribeModel
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try full multimodal (audio + text via LLaMA 3.2-3B) first
        features = ["audio", "text"]
        try:
            from huggingface_hub import model_info
            model_info("meta-llama/Llama-3.2-3B")
            logger.info("LLaMA 3.2-3B access confirmed — loading full multimodal TRIBE")
        except Exception:
            logger.warning("No access to meta-llama/Llama-3.2-3B — falling back to audio-only")
            features = ["audio"]

        config_update = {
            "data.features_to_use": features,
            "data.audio_feature.device": device,
            "data.text_feature.device": device,
            "data.num_workers": 2,
            "data.batch_size": 1,
        }
        logger.info(f"Running TRIBE with {features} on {device}")

        model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder="./cache",
            device="auto",
            config_update=config_update,
        )
        tribe_model = model
        model_mode = "tribe"
        feature_str = "audio + text (LLaMA 3.2-3B)" if "text" in features else "audio only (Wav2Vec-BERT)"
        logger.info(f"TRIBE v2 loaded successfully! Features: {feature_str}")
        return True
    except Exception as e:
        logger.warning(f"Could not load TRIBE v2: {e}")
        logger.info("TRIBE unavailable — brain activation analysis will not be available.")
        model_mode = "unavailable"
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


async def generate_summary(text: str) -> dict:
    """Generate a text-only summary via LLM. No activation values — those come only from TRIBE."""
    prompt = f"""Summarize this journal entry in under 15 words and identify the single strongest emotion or cognitive state present.
Return ONLY a JSON object with exactly two keys: "summary" and "dominant".

Journal entry: "{text}"

Return ONLY valid JSON, no markdown, no explanation."""

    try:
        raw = await llm_chat(prompt, max_tokens=100)
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return {
            "summary": parsed.get("summary", text[:60] + "..."),
            "dominant": parsed.get("dominant", ""),
        }
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        return {"summary": text[:60] + "...", "dominant": ""}


# ─── FastAPI App ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    logger.info("=" * 56)
    logger.info("  NeurJournal — Real-time Brain Activation Platform")
    logger.info("=" * 56)
    load_emotion_model()
    load_tribe_model()
    for d in DOMAIN_REGISTRY.values():
        db.ensure_domain(d["id"], d["name"], d.get("description", ""), d)
    elapsed = time.time() - t0
    logger.info("-" * 56)
    logger.info(f"  Models loaded in {elapsed:.1f}s")
    logger.info(f"  TRIBE: {model_mode} | Emotion: {'yes' if emotion_classifier else 'no'}")
    logger.info(f"  Memory: {len(memory_store.memories)} entries")
    logger.info(f"  Domains: {', '.join(DOMAIN_REGISTRY.keys())}")
    logger.info(f"  Server: http://localhost:8420")
    logger.info("=" * 56)
    yield


app = FastAPI(title="NeurJournal API", version="2.0.0", lifespan=lifespan)

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
    mode: str  # "tribe" only — no fake fallback
    regions: dict
    vertex_activations: list[float] | None = None
    summary: str | None = None
    dominant: str | None = None
    n_vertices: int | None = None
    n_timesteps: int | None = None
    raw_stats: dict | None = None
    processing_time_ms: float = 0


@app.get("/health")
async def health():
    features = []
    if tribe_model is not None:
        try:
            cfg = tribe_model.cfg if hasattr(tribe_model, 'cfg') else None
            if cfg:
                features = list(cfg.get("data", {}).get("features_to_use", []))
        except Exception:
            pass
    return {
        "status": "ok",
        "model_mode": model_mode,
        "features": features,
        "has_emotion": emotion_classifier is not None,
        "platform": True,
        "domains": list_domain_configs(),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if model_mode != "tribe" or tribe_model is None:
        raise HTTPException(
            status_code=503,
            detail="TRIBE model is not loaded. Brain activation analysis requires the research-grade model.",
        )

    start = time.time()

    try:
        result = predict_with_tribe(req.text)
    except Exception as e:
        logger.error(f"TRIBE prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"TRIBE prediction failed: {e}")

    try:
        summary_data = await generate_summary(req.text)
        result["summary"] = summary_data.get("summary")
        result["dominant"] = summary_data.get("dominant")
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        result["summary"] = req.text[:60] + "..."
        result["dominant"] = ""

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
    domain_id: str = "therapy"
    session_id: str | None = None
    subject_id: str | None = None


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

    domain_prompt = get_system_prompt(req.domain_id)
    generated_answer = ""
    prompt = f"""{domain_prompt}

Given retrieved memories from conversations involving {', '.join(speakers) if speakers else 'participants'}, answer the question concisely and accurately.

Retrieved memories:
{memory_context}

Question: {req.query}

Answer in 1-3 sentences based ONLY on the memories above. If the memories don't contain enough information, say so."""

    try:
        generated_answer = await llm_chat(prompt, max_tokens=300)
    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}")
        generated_answer = f"Based on memories: {retrieved[0]['text']}" if retrieved else "No relevant memories found."

    tribe_result = None
    if req.run_tribe and generated_answer and model_mode == "tribe" and tribe_model is not None:
        try:
            tribe_result = predict_with_tribe(generated_answer)
        except Exception as e:
            logger.error(f"TRIBE on memory answer failed: {e}")

    emotion_data = None
    if emotion_classifier and generated_answer:
        try:
            emo_results = emotion_classifier(generated_answer[:512])
            emotions = [{"label": r["label"], "score": round(r["score"], 4)} for r in emo_results[0]]
            emotion_data = {"emotions": emotions, "dominant": max(emotions, key=lambda x: x["score"])["label"]}
        except Exception:
            pass

    elapsed = (time.time() - start) * 1000

    if req.session_id and req.subject_id:
        try:
            db.save_activation(
                session_id=req.session_id,
                subject_id=req.subject_id,
                query=req.query,
                generated_answer=generated_answer,
                regions=tribe_result.get("regions", {}) if tribe_result else {},
                vertex_activations=tribe_result.get("vertex_activations") if tribe_result else None,
                emotion=emotion_data or {},
                processing_time_ms=round(elapsed, 1),
                model_mode=tribe_result.get("mode", "unknown") if tribe_result else "none",
            )
        except Exception as e:
            logger.warning(f"Could not persist activation: {e}")

    return {
        "query": req.query,
        "retrieved_memories": retrieved,
        "generated_answer": generated_answer,
        "tribe": tribe_result,
        "emotion": emotion_data,
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


@app.post("/llm_chat")
async def llm_chat_endpoint(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 300)
    try:
        text = await llm_chat(prompt, max_tokens)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emotion")
async def detect_emotion(request: Request):
    body = await request.json()
    text = body.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if emotion_classifier is None:
        return {"emotions": [], "dominant": "neutral"}
    results = emotion_classifier(text[:512])
    emotions = [{"label": r["label"], "score": round(r["score"], 4)} for r in results[0]]
    dominant = max(emotions, key=lambda x: x["score"])["label"]
    return {"emotions": emotions, "dominant": dominant}


# ─── Platform: Domain Endpoints ───


@app.get("/domains")
async def domains_list():
    return {"domains": list_domain_configs()}


@app.get("/domains/{domain_id}")
async def domain_detail(domain_id: str):
    d = get_domain(domain_id)
    if not d:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")
    return d


# ─── Platform: Subject Endpoints ───


class CreateSubjectRequest(BaseModel):
    domain_id: str
    name: str
    metadata: dict = {}


@app.post("/subjects")
async def create_subject(req: CreateSubjectRequest):
    d = get_domain(req.domain_id)
    if not d:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {req.domain_id}")
    return db.create_subject(req.domain_id, req.name, req.metadata)


@app.get("/subjects")
async def list_subjects(domain_id: str = None):
    return {"subjects": db.list_subjects(domain_id)}


@app.get("/subjects/{subject_id}/history")
async def subject_history(subject_id: str, limit: int = 50):
    return {"history": db.get_subject_history(subject_id, limit)}


@app.get("/subjects/{subject_id}/trends")
async def subject_trends(subject_id: str):
    return db.get_subject_trends(subject_id)


# ─── Platform: Session Endpoints ───


class CreateSessionRequest(BaseModel):
    subject_id: str
    domain_id: str
    metadata: dict = {}


@app.post("/sessions")
async def create_session(req: CreateSessionRequest):
    return db.create_session(req.subject_id, req.domain_id, req.metadata)


# ─── Platform: Save Activation (called after analyze/memory query) ───


class SaveActivationRequest(BaseModel):
    session_id: str
    subject_id: str
    query: str
    generated_answer: str = ""
    regions: dict = {}
    vertex_activations: list[float] | None = None
    emotion: dict = {}
    processing_time_ms: float = 0
    model_mode: str = "tribe"


@app.post("/activations")
async def save_activation(req: SaveActivationRequest):
    return db.save_activation(
        session_id=req.session_id,
        subject_id=req.subject_id,
        query=req.query,
        generated_answer=req.generated_answer,
        regions=req.regions,
        vertex_activations=req.vertex_activations,
        emotion=req.emotion,
        processing_time_ms=req.processing_time_ms,
        model_mode=req.model_mode,
    )


# ─── Cognitive Weather ───


@app.get("/cognitive-weather")
async def cognitive_weather(subject_id: str = None):
    """Aggregate recent activations into a cognitive weather report."""
    history = db.get_subject_history(subject_id, limit=20) if subject_id else []
    if len(history) < 2:
        return {"available": False}

    region_data: dict[str, dict] = {}
    emotions: dict[str, int] = {}
    for h in history:
        for rk, rv in h.get("regions", {}).items():
            if rk not in region_data:
                region_data[rk] = {"label": rv.get("label", rk), "values": []}
            region_data[rk]["values"].append(rv.get("activation", 0))
        emo = h.get("emotion", {}).get("dominant")
        if emo:
            emotions[emo] = emotions.get(emo, 0) + 1

    metrics = {}
    for rk, data in region_data.items():
        vals = data["values"]
        avg = sum(vals) / len(vals) if vals else 0
        recent = vals[:3]
        recent_avg = sum(recent) / len(recent) if recent else 0
        metrics[rk] = {
            "label": data["label"],
            "average": round(avg, 4),
            "recent": round(recent_avg, 4),
            "trend": round(recent_avg - avg, 4),
        }

    dominant = max(metrics.items(), key=lambda x: x[1]["average"]) if metrics else None
    trends_sorted = sorted(metrics.items(), key=lambda x: abs(x[1]["trend"]), reverse=True)
    rising = [{"key": k, **v} for k, v in trends_sorted if v["trend"] > 0.02][:2]
    falling = [{"key": k, **v} for k, v in trends_sorted if v["trend"] < -0.02][:2]

    return {
        "available": True,
        "total_entries": len(history),
        "dominant": {"key": dominant[0], **dominant[1]} if dominant else None,
        "rising": rising,
        "falling": falling,
        "top_emotions": dict(sorted(emotions.items(), key=lambda x: -x[1])[:3]),
        "metrics": metrics,
    }


# ─── Platform: File Upload & Custom Ingestion ───


@app.post("/upload")
async def upload_content(
    file: UploadFile = File(...),
    domain_id: str = Form("therapy"),
    subject_id: str = Form(None),
):
    d = get_domain(domain_id)
    if not d:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {domain_id}")

    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    record = db.save_upload(
        domain_id=domain_id,
        filename=file.filename,
        content=text,
        subject_id=subject_id,
    )

    n_chunks = 0
    if file.filename.endswith(".json"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if "text" in item:
                        memory_store.memories.append(item)
                        n_chunks += 1
                memory_store._build_index()
        except json.JSONDecodeError:
            pass
    elif file.filename.endswith(".txt") or file.filename.endswith(".csv"):
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in lines:
            memory_store.memories.append({
                "text": line,
                "speaker": subject_id or "unknown",
                "session": 0,
                "date": "",
                "type": "uploaded",
            })
            n_chunks += 1
        if n_chunks:
            memory_store._build_index()

    return {
        "id": record["id"],
        "filename": file.filename,
        "chunks_ingested": n_chunks,
        "domain_id": domain_id,
    }


# ─── Static Files ───


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
    backend_dir = str(Path(__file__).resolve().parent)
    dev_mode = os.environ.get("DEV", "").lower() in ("1", "true", "yes")
    if dev_mode:
        logger.warning(
            "DEV mode: hot reload ON — TRIBE & emotion models will re-load on every "
            "file change. Run without DEV=1 for production to keep models resident."
        )
        uvicorn.run(
            "backend.server:app",
            host="0.0.0.0",
            port=8420,
            log_level="info",
            reload=True,
            reload_dirs=[backend_dir],
            reload_includes=["*.py"],
            reload_excludes=["*.db", "*.db-wal", "*.db-shm", "__pycache__/*"],
        )
    else:
        # Pass app object directly — avoids uvicorn re-importing the module
        # which would create duplicate MemoryStore / SentenceTransformer instances
        uvicorn.run(app, host="0.0.0.0", port=8420, log_level="info")
