import { useState, useEffect, useRef, useCallback } from "react";

// ─── Config ───
const API_BASE = "http://localhost:8420";

const BRAIN_REGIONS = {
  prefrontal_cortex: { label: "Prefrontal Cortex", x: 28, y: 28, r: 22, functions: ["reasoning", "planning", "calm", "decision_making"] },
  anterior_cingulate: { label: "Anterior Cingulate", x: 40, y: 35, r: 17, functions: ["conflict", "motivation", "sadness", "error_detection"] },
  insula: { label: "Insula", x: 58, y: 45, r: 16, functions: ["disgust", "empathy", "self_awareness", "interoception"] },
  temporal_lobe: { label: "Temporal Lobe", x: 72, y: 50, r: 20, functions: ["language", "comprehension", "social", "semantics"] },
  amygdala: { label: "Amygdala", x: 50, y: 62, r: 18, functions: ["fear", "anxiety", "anger", "threat_detection"] },
  hippocampus: { label: "Hippocampus", x: 45, y: 68, r: 15, functions: ["memory", "nostalgia", "learning", "spatial_memory"] },
  nucleus_accumbens: { label: "Nucleus Accumbens", x: 48, y: 52, r: 14, functions: ["joy", "reward", "excitement", "motivation"] },
  parietal_lobe: { label: "Parietal Lobe", x: 55, y: 22, r: 20, functions: ["attention", "spatial", "integration", "body_awareness"] },
  occipital_lobe: { label: "Visual Cortex", x: 70, y: 30, r: 17, functions: ["imagery", "visualization", "creativity", "visual_processing"] },
  motor_cortex: { label: "Motor Cortex", x: 38, y: 18, r: 16, functions: ["action", "restlessness", "energy", "motor_planning"] },
};

const FUNCTION_COLORS = {
  fear: "#ef4444", anxiety: "#f97316", anger: "#dc2626", threat_detection: "#b91c1c",
  joy: "#22c55e", reward: "#10b981", excitement: "#06b6d4", motivation: "#14b8a6",
  sadness: "#6366f1", calm: "#8b5cf6", empathy: "#ec4899", self_awareness: "#0ea5e9",
  nostalgia: "#f59e0b", creativity: "#a855f7", reasoning: "#3b82f6", planning: "#2563eb",
  memory: "#d97706", learning: "#0891b2", conflict: "#e11d48", decision_making: "#4f46e5",
  language: "#7c3aed", comprehension: "#4f46e5", social: "#db2777", semantics: "#6d28d9",
  attention: "#0ea5e9", spatial: "#6366f1", integration: "#8b5cf6", body_awareness: "#7c3aed",
  imagery: "#c026d3", visualization: "#a21caf", visual_processing: "#9333ea",
  action: "#059669", restlessness: "#f43f5e", energy: "#eab308", motor_planning: "#16a34a",
  disgust: "#84cc16", interoception: "#06b6d4", error_detection: "#f43f5e", spatial_memory: "#d97706",
};

const FUNCTION_ICONS = {
  joy: "✦", excitement: "⚡", reward: "◆", fear: "▲", anxiety: "◇",
  anger: "■", sadness: "●", calm: "○", empathy: "♡", nostalgia: "◈",
  motivation: "→", creativity: "✧", reasoning: "△", planning: "▷",
  memory: "◉", learning: "▽", conflict: "✕", language: "¶",
  comprehension: "§", social: "※", attention: "◐", action: "►",
  energy: "↯", default: "●",
};

function formatDate(ts) {
  const diff = Date.now() - ts;
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return `${Math.floor(diff / 86400000)}d ago`;
}

function getRegionColor(region, regionData) {
  if (!regionData || !region) return "#334155";
  const activation = regionData.activation || 0;
  if (activation < 0.1) return "#334155";
  return FUNCTION_COLORS[region.functions[0]] || "#6366f1";
}

// ─── Brain Map ───
function BrainMap({ regionActivations, isAnalyzing, size = 340 }) {
  const [pulse, setPulse] = useState(0);
  useEffect(() => { const i = setInterval(() => setPulse(p => (p + 1) % 360), 50); return () => clearInterval(i); }, []);

  return (
    <div style={{ position: "relative", width: size, height: size, margin: "0 auto" }}>
      <svg viewBox="0 0 100 100" width={size} height={size} style={{ overflow: "visible" }}>
        <defs>
          {Object.entries(BRAIN_REGIONS).map(([key, region]) => {
            const data = regionActivations?.[key];
            const a = data?.activation || 0;
            const c = getRegionColor(region, data);
            return <radialGradient key={`g-${key}`} id={`g-${key}`}><stop offset="0%" stopColor={c} stopOpacity={a * 0.9} /><stop offset="60%" stopColor={c} stopOpacity={a * 0.4} /><stop offset="100%" stopColor={c} stopOpacity={0} /></radialGradient>;
          })}
          <radialGradient id="bb"><stop offset="0%" stopColor="#1e293b" /><stop offset="85%" stopColor="#0f172a" /><stop offset="100%" stopColor="#020617" /></radialGradient>
          <filter id="gl"><feGaussianBlur stdDeviation="2" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        <ellipse cx="50" cy="45" rx="38" ry="40" fill="url(#bb)" stroke="#1e293b" strokeWidth="0.5" />
        <path d="M 50 5 Q 30 8 25 20 Q 18 35 15 50 Q 13 65 20 75 Q 30 88 50 90 Q 70 88 80 75 Q 87 65 85 50 Q 82 35 75 20 Q 70 8 50 5" fill="none" stroke="#334155" strokeWidth="0.3" opacity="0.5" />
        <path d="M 50 8 L 50 88" stroke="#1e293b" strokeWidth="0.3" strokeDasharray="1 2" opacity="0.4" />
        {Object.entries(BRAIN_REGIONS).map(([key, region]) => {
          const data = regionActivations?.[key]; const a = data?.activation || 0; const c = getRegionColor(region, data);
          const p = Math.sin(((pulse + region.x * 3) * Math.PI) / 180) * 0.15;
          const r = region.r * (0.8 + a * 0.5 + (a > 0.3 ? p : 0));
          return <g key={key}>{a > 0.1 ? (<><circle cx={region.x} cy={region.y} r={r * 1.3} fill={`url(#g-${key})`} filter="url(#gl)" opacity={0.6} /><circle cx={region.x} cy={region.y} r={r} fill={`url(#g-${key})`} opacity={0.8} /><circle cx={region.x} cy={region.y} r={3} fill={c} opacity={a} /></>) : (<circle cx={region.x} cy={region.y} r={2} fill="#334155" opacity={0.3} />)}</g>;
        })}
        {Object.entries(BRAIN_REGIONS).map(([k1, r1], i) => {
          const a1 = regionActivations?.[k1]?.activation || 0;
          return Object.entries(BRAIN_REGIONS).slice(i + 1).map(([k2, r2]) => {
            const a2 = regionActivations?.[k2]?.activation || 0;
            return (a1 > 0.4 && a2 > 0.4) ? <line key={`${k1}-${k2}`} x1={r1.x} y1={r1.y} x2={r2.x} y2={r2.y} stroke={getRegionColor(r1, regionActivations?.[k1])} strokeWidth={0.3} opacity={Math.min(a1, a2) * 0.3} strokeDasharray="1 2" /> : null;
          });
        })}
        {isAnalyzing && <line x1="12" x2="88" y1={5 + ((pulse * 2) % 85)} y2={5 + ((pulse * 2) % 85)} stroke="#06b6d4" strokeWidth="0.5" opacity={0.4}><animate attributeName="opacity" values="0.4;0.1;0.4" dur="1s" repeatCount="indefinite" /></line>}
      </svg>
    </div>
  );
}

function EmotionBar({ name, value }) {
  const color = FUNCTION_COLORS[name] || "#6366f1";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
      <span style={{ fontSize: 11, color: "#94a3b8", width: 100, textAlign: "right", fontFamily: "var(--mono)", textTransform: "capitalize" }}>{FUNCTION_ICONS[name] || "●"} {name.replace(/_/g, " ")}</span>
      <div style={{ flex: 1, height: 6, background: "#0f172a", borderRadius: 3, overflow: "hidden" }}><div style={{ width: `${value * 100}%`, height: "100%", background: `linear-gradient(90deg, ${color}88, ${color})`, borderRadius: 3, transition: "width 0.8s cubic-bezier(0.16, 1, 0.3, 1)" }} /></div>
      <span style={{ fontSize: 10, color: "#64748b", width: 32, fontFamily: "var(--mono)" }}>{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

function computePatterns(entries) {
  if (entries.length < 2) return [];
  const patterns = [];
  const regionTotals = {};
  entries.forEach(e => { if (!e.regions) return; Object.entries(e.regions).forEach(([k, v]) => { if (!regionTotals[k]) regionTotals[k] = { total: 0, count: 0, label: v.label }; regionTotals[k].total += v.activation; regionTotals[k].count += 1; }); });
  let topR = null, topA = 0;
  for (const [k, v] of Object.entries(regionTotals)) { const avg = v.total / entries.length; if (avg > topA) { topA = avg; topR = { key: k, ...v, avg }; } }
  if (topR && topA > 0.3) { const rd = BRAIN_REGIONS[topR.key]; const c = FUNCTION_COLORS[rd?.functions[0]] || "#6366f1"; patterns.push({ title: `${topR.label} is your baseline`, desc: `Across ${entries.length} entries, this region activates at ${(topA * 100).toFixed(0)}% avg.`, color: c, icon: "◎" }); }
  if (entries.length >= 3) {
    const recent = entries.slice(0, Math.ceil(entries.length / 2)), older = entries.slice(Math.ceil(entries.length / 2));
    let bigR = null, bigV = 0;
    for (const k of Object.keys(regionTotals)) { const ra = recent.reduce((s, e) => s + (e.regions?.[k]?.activation || 0), 0) / recent.length; const oa = older.reduce((s, e) => s + (e.regions?.[k]?.activation || 0), 0) / older.length; const d = ra - oa; if (d > bigV) { bigV = d; bigR = { key: k, label: regionTotals[k].label }; } }
    if (bigR && bigV > 0.08) { const rd = BRAIN_REGIONS[bigR.key]; patterns.push({ title: `Rising ${bigR.label}`, desc: `Up ${(bigV * 100).toFixed(0)}% recently.`, color: FUNCTION_COLORS[rd?.functions[0]] || "#6366f1", icon: "↗" }); }
  }
  const coAct = {};
  entries.forEach(e => { if (!e.regions) return; const active = Object.entries(e.regions).filter(([_, v]) => v.activation > 0.5); for (let i = 0; i < active.length; i++) for (let j = i + 1; j < active.length; j++) { const p = [active[i][1].label, active[j][1].label].sort().join(" ↔ "); coAct[p] = (coAct[p] || 0) + 1; } });
  const tp2 = Object.entries(coAct).sort((a, b) => b[1] - a[1])[0];
  if (tp2 && tp2[1] >= 2) patterns.push({ title: "Neural coupling", desc: `${tp2[0]} co-activate in ${tp2[1]}/${entries.length} entries.`, color: "#06b6d4", icon: "⟷" });
  return patterns;
}

const SAMPLES = [
  { id: "s1", text: "Had an amazing breakthrough at work today. Finally figured out the architecture for the agent system.", timestamp: Date.now() - 86400000 * 3, summary: "Work breakthrough", dominant: "joy", mode: "sample", regions: { prefrontal_cortex: { label: "Prefrontal Cortex", activation: 0.7 }, nucleus_accumbens: { label: "Nucleus Accumbens", activation: 0.9 }, temporal_lobe: { label: "Temporal Lobe", activation: 0.5 }, parietal_lobe: { label: "Parietal Lobe", activation: 0.4 }, motor_cortex: { label: "Motor Cortex", activation: 0.5 }, anterior_cingulate: { label: "Anterior Cingulate", activation: 0.3 }, insula: { label: "Insula", activation: 0.2 }, amygdala: { label: "Amygdala", activation: 0.1 }, hippocampus: { label: "Hippocampus", activation: 0.3 }, occipital_lobe: { label: "Visual Cortex", activation: 0.4 } } },
  { id: "s2", text: "Couldn't sleep last night. Kept thinking about whether I'm making the right career moves.", timestamp: Date.now() - 86400000 * 2, summary: "Career anxiety", dominant: "anxiety", mode: "sample", regions: { amygdala: { label: "Amygdala", activation: 0.85 }, anterior_cingulate: { label: "Anterior Cingulate", activation: 0.7 }, insula: { label: "Insula", activation: 0.8 }, prefrontal_cortex: { label: "Prefrontal Cortex", activation: 0.4 }, temporal_lobe: { label: "Temporal Lobe", activation: 0.3 }, hippocampus: { label: "Hippocampus", activation: 0.4 }, nucleus_accumbens: { label: "Nucleus Accumbens", activation: 0.1 }, parietal_lobe: { label: "Parietal Lobe", activation: 0.2 }, occipital_lobe: { label: "Visual Cortex", activation: 0.15 }, motor_cortex: { label: "Motor Cortex", activation: 0.3 } } },
  { id: "s3", text: "Long walk in the park with an old friend. Talked about college days and laughed about old times.", timestamp: Date.now() - 86400000, summary: "Nostalgic walk", dominant: "nostalgia", mode: "sample", regions: { hippocampus: { label: "Hippocampus", activation: 0.9 }, temporal_lobe: { label: "Temporal Lobe", activation: 0.8 }, nucleus_accumbens: { label: "Nucleus Accumbens", activation: 0.65 }, insula: { label: "Insula", activation: 0.6 }, prefrontal_cortex: { label: "Prefrontal Cortex", activation: 0.5 }, anterior_cingulate: { label: "Anterior Cingulate", activation: 0.4 }, amygdala: { label: "Amygdala", activation: 0.2 }, parietal_lobe: { label: "Parietal Lobe", activation: 0.3 }, occipital_lobe: { label: "Visual Cortex", activation: 0.5 }, motor_cortex: { label: "Motor Cortex", activation: 0.4 } } },
];

// ─── Main App ───
export default function NeuroJournal() {
  const [entries, setEntries] = useState(SAMPLES);
  const [inputText, setInputText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedEntry, setSelectedEntry] = useState(null);
  const [activeView, setActiveView] = useState("journal");
  const [backend, setBackend] = useState({ status: "checking", model_mode: "unknown" });

  useEffect(() => { fetch(`${API_BASE}/health`).then(r => r.json()).then(setBackend).catch(() => setBackend({ status: "offline", model_mode: "browser" })); }, []);

  const currentRegions = selectedEntry?.regions || entries[0]?.regions || {};
  const patterns = computePatterns(entries);

  const allFuncs = [];
  if (selectedEntry?.raw_emotions) { Object.entries(selectedEntry.raw_emotions).forEach(([k, v]) => allFuncs.push({ name: k, value: v })); }
  else if (selectedEntry?.regions) { Object.entries(selectedEntry.regions).forEach(([_, r]) => { (r.functions || [r.label]).forEach(f => typeof f === "string" && allFuncs.push({ name: f, value: r.activation })); }); }
  const seen = new Set();
  const uniqFuncs = allFuncs.sort((a, b) => b.value - a.value).filter(f => f.value > 0.05 && !seen.has(f.name) && seen.add(f.name));

  const handleSubmit = useCallback(async () => {
    if (!inputText.trim() || isAnalyzing) return;
    setIsAnalyzing(true);
    try {
      let result;
      if (backend.status === "ok") {
        const resp = await fetch(`${API_BASE}/analyze`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: inputText }) });
        result = await resp.json();
      } else {
        const allF = new Set(); Object.values(BRAIN_REGIONS).forEach(r => r.functions.forEach(f => allF.add(f)));
        const resp = await fetch("https://api.anthropic.com/v1/messages", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: "claude-sonnet-4-20250514", max_tokens: 1000, messages: [{ role: "user", content: `Analyze this journal entry. Return ONLY JSON mapping these to 0.0-1.0: ${[...allF].join(", ")}. Include "summary" (<15 words) and "dominant". Only >0.1. Entry: "${inputText}"` }] }) });
        const data = await resp.json();
        const parsed = JSON.parse(data.content[0].text.replace(/```json|```/g, "").trim());
        const summary = parsed.summary; delete parsed.summary; const dominant = parsed.dominant; delete parsed.dominant;
        const regions = {};
        for (const [rk, rv] of Object.entries(BRAIN_REGIONS)) { let mx = 0; rv.functions.forEach(f => { if (parsed[f] > mx) mx = parsed[f]; }); regions[rk] = { label: rv.label, activation: mx, functions: rv.functions }; }
        result = { mode: "browser", regions, summary, dominant, raw_emotions: parsed };
      }
      const newEntry = { id: `e-${Date.now()}`, text: inputText, timestamp: Date.now(), ...result };
      setEntries(prev => [newEntry, ...prev]);
      setSelectedEntry(newEntry);
      setInputText("");
    } catch (err) { console.error("Analysis failed:", err); }
    setIsAnalyzing(false);
  }, [inputText, isAnalyzing, backend]);

  const modeLabel = backend.model_mode === "tribe" ? "TRIBE v2 LIVE" : backend.status === "ok" ? "LOADING" : backend.status === "offline" ? "OFFLINE" : "...";
  const modeColor = backend.model_mode === "tribe" ? "#22c55e" : backend.status === "ok" ? "#f59e0b" : "#ef4444";

  return (
    <div style={{ minHeight: "100vh", background: "#020617", color: "#e2e8f0", fontFamily: "var(--body)", "--mono": "'JetBrains Mono', monospace", "--body": "'Inter', system-ui, sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet" />
      <div style={{ position: "fixed", inset: 0, opacity: 0.3, pointerEvents: "none", background: `radial-gradient(ellipse at 20% 50%, ${modeColor}12 0%, transparent 50%)` }} />
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "24px 20px", position: "relative", zIndex: 1 }}>
        <div style={{ marginBottom: 28 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: isAnalyzing ? "#f59e0b" : modeColor, boxShadow: `0 0 12px ${modeColor}66`, animation: isAnalyzing ? "pulse 1s infinite" : "none" }} />
            <h1 style={{ fontSize: 22, fontWeight: 600, fontFamily: "var(--mono)", color: "#f8fafc", letterSpacing: "-0.5px", margin: 0 }}>neurojournal</h1>
            <span style={{ fontSize: 9, padding: "2px 8px", background: `${modeColor}18`, color: modeColor, borderRadius: 4, border: `1px solid ${modeColor}44`, fontFamily: "var(--mono)", fontWeight: 600 }}>{modeLabel}</span>
          </div>
          <p style={{ fontSize: 12, color: "#475569", margin: 0, fontFamily: "var(--mono)" }}>emotional activation mapping · meta TRIBE v2 · memory-augmented journaling</p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: 20, alignItems: "start" }}>
          <div>
            <div style={{ background: "linear-gradient(135deg, #0f172a, #0a0f1e)", border: "1px solid #1e293b", borderRadius: 14, padding: 18, marginBottom: 16 }}>
              <textarea value={inputText} onChange={e => setInputText(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && e.metaKey) handleSubmit(); }} placeholder="Write freely — your neural activations will be mapped through TRIBE v2..." rows={4} style={{ width: "100%", background: "transparent", border: "none", outline: "none", color: "#e2e8f0", fontSize: 14, fontFamily: "var(--body)", resize: "vertical", lineHeight: 1.7, boxSizing: "border-box" }} />
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 10 }}>
                <span style={{ fontSize: 10, color: "#334155", fontFamily: "var(--mono)" }}>⌘+Enter</span>
                <button onClick={handleSubmit} disabled={!inputText.trim() || isAnalyzing} style={{ background: isAnalyzing ? "linear-gradient(135deg, #f59e0b, #d97706)" : "linear-gradient(135deg, #6366f1, #4f46e5)", color: "#fff", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 12, fontFamily: "var(--mono)", fontWeight: 500, cursor: isAnalyzing ? "wait" : "pointer", opacity: !inputText.trim() ? 0.4 : 1, transition: "all 0.3s" }}>
                  {isAnalyzing ? "◎ Scanning neural activations..." : "▶ Analyze"}
                </button>
              </div>
            </div>
            <div style={{ display: "flex", gap: 2, marginBottom: 14, background: "#0f172a", borderRadius: 8, padding: 3, border: "1px solid #1e293b" }}>
              {["journal", "patterns"].map(k => <button key={k} onClick={() => setActiveView(k)} style={{ flex: 1, padding: "7px 0", border: "none", borderRadius: 6, background: activeView === k ? "#1e293b" : "transparent", color: activeView === k ? "#e2e8f0" : "#475569", fontSize: 11, fontFamily: "var(--mono)", fontWeight: 500, cursor: "pointer" }}>{k === "patterns" ? `Patterns ${patterns.length > 0 ? `(${patterns.length})` : ""}` : "Journal"}</button>)}
            </div>
            {activeView === "journal" && entries.map(entry => {
              const sel = selectedEntry?.id === entry.id;
              const tr = entry.regions ? Object.entries(entry.regions).sort((a, b) => b[1].activation - a[1].activation)[0] : null;
              const c = tr ? getRegionColor(BRAIN_REGIONS[tr[0]], tr[1]) : "#6366f1";
              return (
                <div key={entry.id} onClick={() => setSelectedEntry(entry)} style={{ background: sel ? `linear-gradient(135deg, ${c}11, ${c}06)` : "#0f172a", border: `1px solid ${sel ? `${c}44` : "#1e293b"}`, borderRadius: 12, padding: 14, marginBottom: 8, cursor: "pointer", borderLeft: `3px solid ${c}${sel ? "" : "66"}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                      <span style={{ fontSize: 12, fontWeight: 500, color: c, fontFamily: "var(--mono)" }}>{entry.dominant || tr?.[1]?.label || ""}</span>
                      {entry.mode && entry.mode !== "sample" && <span style={{ fontSize: 8, padding: "1px 5px", borderRadius: 3, background: entry.mode === "tribe" ? "#22c55e18" : "#f59e0b18", color: entry.mode === "tribe" ? "#22c55e" : "#f59e0b", fontFamily: "var(--mono)" }}>{entry.mode === "tribe" ? "TRIBE" : "API"}</span>}
                    </div>
                    <span style={{ fontSize: 10, color: "#475569", fontFamily: "var(--mono)" }}>{formatDate(entry.timestamp)}</span>
                  </div>
                  <p style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.5, margin: "0 0 6px" }}>{entry.text.length > 140 ? entry.text.slice(0, 140) + "…" : entry.text}</p>
                  <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                    {entry.regions && Object.entries(entry.regions).sort((a, b) => b[1].activation - a[1].activation).slice(0, 4).map(([k, v]) => (
                      <span key={k} style={{ fontSize: 9, padding: "2px 6px", borderRadius: 4, background: `${getRegionColor(BRAIN_REGIONS[k], v)}18`, color: getRegionColor(BRAIN_REGIONS[k], v), fontFamily: "var(--mono)" }}>{v.label} {(v.activation * 100).toFixed(0)}%</span>
                    ))}
                  </div>
                </div>
              );
            })}
            {activeView === "patterns" && (patterns.length === 0 ? <div style={{ padding: 40, textAlign: "center" }}><p style={{ color: "#475569" }}>Add more entries to detect patterns.</p></div> : patterns.map((p, i) => (
              <div key={i} style={{ background: `linear-gradient(135deg, ${p.color}11, ${p.color}06)`, border: `1px solid ${p.color}33`, borderRadius: 10, padding: "12px 14px", marginBottom: 8 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: p.color, marginBottom: 4, fontFamily: "var(--mono)" }}>{p.icon} {p.title}</div>
                <div style={{ fontSize: 11, color: "#94a3b8", lineHeight: 1.5 }}>{p.desc}</div>
              </div>
            )))}
          </div>

          <div>
            <div style={{ background: "linear-gradient(135deg, #0f172a, #080d18)", border: "1px solid #1e293b", borderRadius: 14, padding: 18, position: "sticky", top: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <span style={{ fontSize: 11, fontFamily: "var(--mono)", color: "#475569", fontWeight: 500 }}>CORTICAL ACTIVATION MAP</span>
                {selectedEntry?.n_vertices && <span style={{ fontSize: 9, color: "#334155", fontFamily: "var(--mono)" }}>{selectedEntry.n_vertices}v</span>}
              </div>
              <BrainMap regionActivations={currentRegions} isAnalyzing={isAnalyzing} size={320} />
              {selectedEntry?.summary && <p style={{ fontSize: 11, color: "#64748b", textAlign: "center", margin: "10px 0 0", fontFamily: "var(--mono)" }}>{selectedEntry.summary}</p>}
              {selectedEntry?.processing_time_ms && <p style={{ fontSize: 9, color: "#1e293b", textAlign: "center", margin: "4px 0", fontFamily: "var(--mono)" }}>{selectedEntry.processing_time_ms.toFixed(0)}ms</p>}
              <div style={{ marginTop: 16, borderTop: "1px solid #1e293b", paddingTop: 14 }}>
                <span style={{ fontSize: 10, fontFamily: "var(--mono)", color: "#475569", fontWeight: 500, display: "block", marginBottom: 10 }}>ACTIVATION SPECTRUM</span>
                {uniqFuncs.slice(0, 8).map(f => <EmotionBar key={f.name} name={f.name} value={f.value} />)}
                {uniqFuncs.length > 8 && <span style={{ fontSize: 10, color: "#334155", fontFamily: "var(--mono)" }}>+{uniqFuncs.length - 8} more</span>}
              </div>
              <div style={{ marginTop: 14, borderTop: "1px solid #1e293b", paddingTop: 12 }}>
                <span style={{ fontSize: 10, fontFamily: "var(--mono)", color: "#475569", fontWeight: 500, display: "block", marginBottom: 8 }}>ACTIVE REGIONS</span>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {Object.entries(currentRegions).filter(([_, v]) => v.activation > 0.3).sort((a, b) => b[1].activation - a[1].activation).map(([k, v]) => {
                    const c = getRegionColor(BRAIN_REGIONS[k], v);
                    return <span key={k} style={{ fontSize: 9, padding: "3px 7px", borderRadius: 4, background: `${c}18`, color: c, fontFamily: "var(--mono)", border: `1px solid ${c}33` }}>{v.label} {(v.activation * 100).toFixed(0)}%</span>;
                  })}
                </div>
              </div>
              <div style={{ marginTop: 14, borderTop: "1px solid #1e293b", paddingTop: 12 }}>
                <span style={{ fontSize: 10, fontFamily: "var(--mono)", color: "#475569", fontWeight: 500, display: "block", marginBottom: 8 }}>TIMELINE</span>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {entries.slice(0, 10).reverse().map(entry => {
                    const sel = selectedEntry?.id === entry.id;
                    const tr = entry.regions ? Object.entries(entry.regions).sort((a, b) => b[1].activation - a[1].activation)[0] : null;
                    const c = tr ? getRegionColor(BRAIN_REGIONS[tr[0]], tr[1]) : "#6366f1";
                    return <button key={entry.id} onClick={() => setSelectedEntry(entry)} title={entry.summary} style={{ background: sel ? c : "transparent", border: `2px solid ${c}`, borderRadius: "50%", width: 28, height: 28, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: sel ? `0 0 12px ${c}66` : "none", fontSize: 10, color: sel ? "#020617" : c }}>{FUNCTION_ICONS[entry.dominant] || "●"}</button>;
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
        <div style={{ textAlign: "center", marginTop: 32, paddingBottom: 20 }}><span style={{ fontSize: 10, color: "#1e293b", fontFamily: "var(--mono)" }}>neurojournal · meta TRIBE v2 brain encoding · founders inc night hacks</span></div>
      </div>
      <style>{`@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } } textarea::placeholder { color: #334155; } ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; } * { box-sizing: border-box; }`}</style>
    </div>
  );
}
