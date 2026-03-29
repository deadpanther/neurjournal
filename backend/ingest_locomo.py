"""
Ingest LoCoMo conversations into a flat memory store (JSON).

Downloads locomo10.json from GitHub, parses conversations 0 (Caroline/Melanie)
and 1 (Jon/Gina), and extracts observations + event_summaries as memory records.

Usage:
    python backend/ingest_locomo.py
"""

import json
import re
import urllib.request
from pathlib import Path

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA_DIR = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "memories.json"
CACHE_PATH = DATA_DIR / "locomo10.json"

CONVERSATIONS_TO_INGEST = [0, 1]


def download_locomo() -> list[dict]:
    if CACHE_PATH.exists():
        print(f"Using cached {CACHE_PATH}")
    else:
        print(f"Downloading locomo10.json ...")
        urllib.request.urlretrieve(LOCOMO_URL, CACHE_PATH)
        print(f"Saved to {CACHE_PATH}")
    with open(CACHE_PATH) as f:
        return json.load(f)


def extract_session_dates(conversation: dict) -> dict[int, str]:
    """Map session number -> date string from session_N_date_time keys."""
    dates = {}
    for key, val in conversation.items():
        m = re.match(r"session_(\d+)_date_time", key)
        if m:
            dates[int(m.group(1))] = val
    return dates


def extract_memories(conv_idx: int, entry: dict) -> list[dict]:
    memories = []
    conversation = entry["conversation"]
    speakers = [conversation["speaker_a"], conversation["speaker_b"]]
    session_dates = extract_session_dates(conversation)

    # --- observations (richest: factual, with evidence dialog IDs) ---
    observations = entry.get("observation", {})
    for obs_key, obs_val in observations.items():
        m = re.match(r"session_(\d+)_observation", obs_key)
        if not m:
            continue
        session_num = int(m.group(1))
        date = session_dates.get(session_num, "")

        for speaker in speakers:
            speaker_obs = obs_val.get(speaker, [])
            for i, item in enumerate(speaker_obs):
                text = item[0] if isinstance(item, list) else item
                evidence = [item[1]] if isinstance(item, list) and len(item) > 1 else []
                memories.append({
                    "id": f"conv{conv_idx}_obs_s{session_num}_{speaker.lower()}_{i}",
                    "text": text,
                    "type": "observation",
                    "speaker": speaker,
                    "session": session_num,
                    "date": date,
                    "conversation_id": conv_idx,
                    "speakers": speakers,
                    "evidence": evidence,
                })

    # --- event_summaries (more concise, per-session) ---
    events = entry.get("event_summary", {})
    for ev_key, ev_val in events.items():
        m = re.match(r"events_session_(\d+)", ev_key)
        if not m:
            continue
        session_num = int(m.group(1))
        date = ev_val.get("date", session_dates.get(session_num, ""))

        for speaker in speakers:
            speaker_events = ev_val.get(speaker, [])
            for i, text in enumerate(speaker_events):
                memories.append({
                    "id": f"conv{conv_idx}_evt_s{session_num}_{speaker.lower()}_{i}",
                    "text": text,
                    "type": "event_summary",
                    "speaker": speaker,
                    "session": session_num,
                    "date": date,
                    "conversation_id": conv_idx,
                    "speakers": speakers,
                    "evidence": [],
                })

    return memories


def main():
    data = download_locomo()
    all_memories = []
    conv_meta = []

    for idx in CONVERSATIONS_TO_INGEST:
        entry = data[idx]
        conv = entry["conversation"]
        speakers = [conv["speaker_a"], conv["speaker_b"]]
        session_dates = extract_session_dates(conv)

        memories = extract_memories(idx, entry)
        all_memories.extend(memories)

        conv_meta.append({
            "conversation_id": idx,
            "speakers": speakers,
            "num_sessions": len(session_dates),
            "num_memories": len(memories),
            "date_range": f"{session_dates.get(1, '?')} — {session_dates.get(max(session_dates.keys()), '?')}" if session_dates else "unknown",
        })

        print(f"Conv {idx} ({speakers[0]} & {speakers[1]}): {len(memories)} memories from {len(session_dates)} sessions")

    # Also include the QA pairs as metadata (not searchable memories, but useful for eval)
    qa_pairs = []
    for idx in CONVERSATIONS_TO_INGEST:
        for qa in data[idx].get("qa", []):
            qa_pairs.append({
                "conversation_id": idx,
                "question": qa["question"],
                "answer": qa.get("answer", ""),
                "category": qa.get("category", 0),
                "evidence": qa.get("evidence", []),
            })

    output = {
        "conversations": conv_meta,
        "memories": all_memories,
        "sample_questions": qa_pairs,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved {len(all_memories)} memories + {len(qa_pairs)} sample questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
