"""
Domain adapter system — makes NeurJournal extensible to any field.

Each domain is a configuration that customizes:
  - Vocabulary / prompts used for queries
  - Region labeling (same brain regions, different context labels)
  - Sample questions
  - Subject terminology (patient vs student vs user vs participant)
  - Memory ingestion format
  - LLM system prompts for answer generation
"""

DOMAIN_REGISTRY: dict[str, dict] = {
    "therapy": {
        "id": "therapy",
        "name": "Clinical Therapy",
        "icon": "🧠",
        "description": "Track neural patterns during therapeutic conversations to aid treatment planning.",
        "subject_label": "Patient",
        "session_label": "Session",
        "query_placeholder": "Ask about the patient's thoughts, emotions, or experiences...",
        "system_prompt": (
            "You are a clinical assistant helping a therapist understand their patient's neural responses. "
            "Synthesize the retrieved memories into a concise, empathetic answer. "
            "Use language appropriate for a clinical setting."
        ),
        "region_labels": {
            "prefrontal": {"label": "Prefrontal Cortex", "context": "Executive function, emotional regulation"},
            "temporal": {"label": "Temporal Lobe", "context": "Language, memory retrieval, auditory processing"},
            "parietal": {"label": "Parietal Lobe", "context": "Sensory integration, spatial awareness"},
            "occipital": {"label": "Occipital Lobe", "context": "Visual processing, mental imagery"},
            "limbic": {"label": "Limbic System", "context": "Emotional processing, fear, reward"},
            "motor": {"label": "Motor Cortex", "context": "Movement planning, procedural memory"},
            "somatosensory": {"label": "Somatosensory Cortex", "context": "Body awareness, physical sensation"},
        },
        "sample_questions": [
            "How is the patient managing stress lately?",
            "What coping mechanisms has the patient developed?",
            "Describe the patient's emotional state during recent sessions.",
        ],
    },
    "education": {
        "id": "education",
        "name": "Learning & Education",
        "icon": "📚",
        "description": "Monitor cognitive engagement during learning to optimize teaching strategies.",
        "subject_label": "Student",
        "session_label": "Lesson",
        "query_placeholder": "Ask about the student's learning, comprehension, or engagement...",
        "system_prompt": (
            "You are an educational neuroscience assistant. Analyze the student's cognitive engagement "
            "based on retrieved learning interactions. Provide insights about comprehension, attention, "
            "and optimal learning strategies."
        ),
        "region_labels": {
            "prefrontal": {"label": "Prefrontal Cortex", "context": "Working memory, problem solving, attention control"},
            "temporal": {"label": "Temporal Lobe", "context": "Language comprehension, semantic memory"},
            "parietal": {"label": "Parietal Lobe", "context": "Mathematical reasoning, spatial visualization"},
            "occipital": {"label": "Occipital Lobe", "context": "Reading, visual learning, diagram processing"},
            "limbic": {"label": "Limbic System", "context": "Motivation, curiosity, learning anxiety"},
            "motor": {"label": "Motor Cortex", "context": "Procedural learning, writing, lab skills"},
            "somatosensory": {"label": "Somatosensory Cortex", "context": "Hands-on learning, kinesthetic memory"},
        },
        "sample_questions": [
            "How engaged is the student during problem-solving tasks?",
            "What learning style shows the strongest neural response?",
            "Where does the student's attention drop during lectures?",
        ],
    },
    "ux_research": {
        "id": "ux_research",
        "name": "UX Research",
        "icon": "🎨",
        "description": "Measure neural responses to user interfaces for evidence-based design decisions.",
        "subject_label": "Participant",
        "session_label": "Test",
        "query_placeholder": "Ask about the participant's reaction to designs, flows, or interactions...",
        "system_prompt": (
            "You are a UX neuroanalysis assistant. Interpret neural activation patterns from user testing "
            "sessions to provide evidence-based design recommendations. Focus on cognitive load, "
            "emotional valence, and attention patterns."
        ),
        "region_labels": {
            "prefrontal": {"label": "Prefrontal Cortex", "context": "Cognitive load, decision fatigue, task planning"},
            "temporal": {"label": "Temporal Lobe", "context": "Label comprehension, auditory feedback processing"},
            "parietal": {"label": "Parietal Lobe", "context": "Navigation, spatial layout processing"},
            "occipital": {"label": "Occipital Lobe", "context": "Visual hierarchy, color/contrast response"},
            "limbic": {"label": "Limbic System", "context": "Frustration, delight, trust signals"},
            "motor": {"label": "Motor Cortex", "context": "Interaction fluency, gesture/click anticipation"},
            "somatosensory": {"label": "Somatosensory Cortex", "context": "Haptic feedback, touch interaction"},
        },
        "sample_questions": [
            "Which screen caused the highest cognitive load?",
            "How did the participant react to the new checkout flow?",
            "What design element triggered the strongest emotional response?",
        ],
    },
    "neuromarketing": {
        "id": "neuromarketing",
        "name": "Neuromarketing",
        "icon": "📊",
        "description": "Analyze consumer neural responses to ads, brands, and products.",
        "subject_label": "Consumer",
        "session_label": "Exposure",
        "query_placeholder": "Ask about the consumer's neural response to brand stimuli...",
        "system_prompt": (
            "You are a neuromarketing analyst. Interpret brain activation patterns in response to "
            "advertising, branding, and product experiences. Provide insights about attention capture, "
            "emotional engagement, memory encoding, and purchase intent signals."
        ),
        "region_labels": {
            "prefrontal": {"label": "Prefrontal Cortex", "context": "Decision making, brand evaluation, purchase intent"},
            "temporal": {"label": "Temporal Lobe", "context": "Slogan recall, jingle memory, narrative processing"},
            "parietal": {"label": "Parietal Lobe", "context": "Price evaluation, quantity perception"},
            "occipital": {"label": "Occipital Lobe", "context": "Logo recognition, packaging appeal, ad visuals"},
            "limbic": {"label": "Limbic System", "context": "Brand emotion, desire, trust/loyalty signals"},
            "motor": {"label": "Motor Cortex", "context": "Product interaction urge, unboxing anticipation"},
            "somatosensory": {"label": "Somatosensory Cortex", "context": "Texture/material response, tactile appeal"},
        },
        "sample_questions": [
            "Which ad creative drove the strongest emotional response?",
            "How does brand A compare to brand B in memory encoding?",
            "What part of the product demo captured the most attention?",
        ],
    },
    "meditation": {
        "id": "meditation",
        "name": "Meditation & Wellness",
        "icon": "🧘",
        "description": "Track mindfulness states and neural changes across meditation practice.",
        "subject_label": "Practitioner",
        "session_label": "Practice",
        "query_placeholder": "Ask about the practitioner's mindfulness states or neural patterns...",
        "system_prompt": (
            "You are a contemplative neuroscience assistant. Analyze neural activation patterns during "
            "meditation and mindfulness practice. Provide insights about attentional states, "
            "emotional regulation progress, and depth of practice."
        ),
        "region_labels": {
            "prefrontal": {"label": "Prefrontal Cortex", "context": "Focused attention, meta-awareness, self-regulation"},
            "temporal": {"label": "Temporal Lobe", "context": "Mantra processing, sound-based meditation response"},
            "parietal": {"label": "Parietal Lobe", "context": "Body scan awareness, self-other boundary dissolution"},
            "occipital": {"label": "Occipital Lobe", "context": "Visualization practice, internal imagery"},
            "limbic": {"label": "Limbic System", "context": "Emotional equanimity, compassion activation, stress reduction"},
            "motor": {"label": "Motor Cortex", "context": "Stillness regulation, breathing control"},
            "somatosensory": {"label": "Somatosensory Cortex", "context": "Interoception, breath awareness, body sensation"},
        },
        "sample_questions": [
            "How deep was the practitioner's focus during today's session?",
            "Compare emotional regulation across the last 5 sessions.",
            "What type of meditation produces the strongest prefrontal activation?",
        ],
    },
    "sports": {
        "id": "sports",
        "name": "Sports Performance",
        "icon": "⚡",
        "description": "Analyze cognitive-motor patterns in athletes for peak performance training.",
        "subject_label": "Athlete",
        "session_label": "Training",
        "query_placeholder": "Ask about the athlete's cognitive-motor patterns or mental state...",
        "system_prompt": (
            "You are a sports neuroscience assistant. Analyze brain activation patterns during athletic "
            "training and competition scenarios. Focus on motor planning, reaction time, "
            "flow states, and performance anxiety."
        ),
        "region_labels": {
            "prefrontal": {"label": "Prefrontal Cortex", "context": "Strategy, decision-making under pressure"},
            "temporal": {"label": "Temporal Lobe", "context": "Coach instruction processing, play calling"},
            "parietal": {"label": "Parietal Lobe", "context": "Spatial awareness, field positioning, trajectory estimation"},
            "occipital": {"label": "Occipital Lobe", "context": "Visual tracking, opponent reading, peripheral vision"},
            "limbic": {"label": "Limbic System", "context": "Competition drive, performance anxiety, flow state entry"},
            "motor": {"label": "Motor Cortex", "context": "Movement execution, muscle memory, reaction patterns"},
            "somatosensory": {"label": "Somatosensory Cortex", "context": "Proprioception, body control, fatigue awareness"},
        },
        "sample_questions": [
            "How does the athlete's focus change under competitive pressure?",
            "What mental patterns correlate with peak performance flow states?",
            "Where does performance anxiety show up in the brain scan?",
        ],
    },
}


def get_domain(domain_id: str) -> dict | None:
    return DOMAIN_REGISTRY.get(domain_id)


def list_domains() -> list[dict]:
    return [
        {"id": d["id"], "name": d["name"], "icon": d["icon"], "description": d["description"]}
        for d in DOMAIN_REGISTRY.values()
    ]


def get_system_prompt(domain_id: str) -> str:
    domain = get_domain(domain_id)
    if domain:
        return domain["system_prompt"]
    return "You are a helpful assistant analyzing brain activation patterns. Synthesize the retrieved context into a clear answer."


def get_region_context(domain_id: str, region_key: str) -> str:
    domain = get_domain(domain_id)
    if domain:
        labels = domain.get("region_labels", {})
        info = labels.get(region_key, {})
        return info.get("context", "")
    return ""
