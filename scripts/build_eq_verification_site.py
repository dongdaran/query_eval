from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


EQ_TYPE_ORDER = [
    "command",
    "full_caption",
    "indirect",
    "key_phrase",
    "question",
    "statement",
]


def temp_value(name: str) -> float:
    return float(name.removeprefix("temp_").replace("_", "."))


def normalize_temp(name: str) -> str:
    return name.removeprefix("temp_").replace("_", ".")


def first_words(text: str, n: int) -> str:
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())
    return " ".join(words[:n])


def parse_session(value: str) -> tuple[str, str]:
    if "=" in value:
        session, label = value.split("=", 1)
        return session.strip(), label.strip()
    return value.strip(), value.strip()


def load_records(root: Path, sessions: list[tuple[str, str]], models: list[str]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any]] = {}
    temps: set[str] = set()
    eq_types: set[str] = set()
    constraints: list[dict[str, str]] = []

    for session_id, constraint_label in sessions:
        session_dir = root / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Missing session directory: {session_dir}")
        constraints.append({"id": session_id, "label": constraint_label})

        for model in models:
            model_dir = session_dir / model
            if not model_dir.exists():
                continue

            for temp_dir in sorted(model_dir.glob("temp_*"), key=lambda p: temp_value(p.name)):
                temp = normalize_temp(temp_dir.name)
                temps.add(temp)

                for file_path in sorted(temp_dir.glob("eq_*.jsonl")):
                    eq_type = file_path.stem.removeprefix("eq_")
                    eq_types.add(eq_type)
                    with file_path.open(encoding="utf-8") as handle:
                        for line_number, line in enumerate(handle, start=1):
                            if not line.strip():
                                continue
                            obj = json.loads(line)
                            audio_id = obj.get("audio_id", "")
                            captions = obj.get("original_captions") or []
                            source_caption = " | ".join(str(c) for c in captions)
                            cases.setdefault(
                                audio_id,
                                {
                                    "audio_id": audio_id,
                                    "dataset": obj.get("dataset", ""),
                                    "dataset_slug": obj.get("dataset_slug", ""),
                                    "source_caption": source_caption,
                                    "captions": captions,
                                },
                            )
                            query = obj.get("generated_query", "")
                            records.append(
                                {
                                    "id": f"{session_id}::{model}::{temp}::{eq_type}::{audio_id}",
                                    "constraint_id": session_id,
                                    "constraint_label": constraint_label,
                                    "model": model,
                                    "temp": temp,
                                    "eq_type": eq_type,
                                    "audio_id": audio_id,
                                    "dataset": obj.get("dataset", ""),
                                    "source_caption": source_caption,
                                    "generated_query": query,
                                    "explanation": obj.get("explanation", ""),
                                    "first_word": first_words(query, 1),
                                    "first_3_words": first_words(query, 3),
                                    "path": str(file_path.relative_to(root)),
                                    "line": line_number,
                                }
                            )

    ordered_eq_types = [typ for typ in EQ_TYPE_ORDER if typ in eq_types]
    ordered_eq_types.extend(sorted(eq_types - set(ordered_eq_types)))
    return {
        "root": str(root),
        "models": models,
        "temps": sorted(temps, key=float),
        "constraints": constraints,
        "eq_types": ordered_eq_types,
        "cases": sorted(cases.values(), key=lambda c: (c["dataset"], c["audio_id"])),
        "records": records,
    }


def render_html(data: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False)
    title = "EQ Recall Case Verification"
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #65717e;
      --line: #d9ded8;
      --accent: #1d6f61;
      --accent-weak: #e4f1ed;
      --gold: #996a00;
      --gold-weak: #fff4cc;
      --bad: #a43b3b;
      --ok: #246b42;
      --shadow: 0 1px 2px rgba(0,0,0,.06), 0 10px 24px rgba(31,41,51,.06);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(247,247,244,.94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }}
    .wrap {{ max-width: 1560px; margin: 0 auto; padding: 18px 22px; }}
    h1 {{ margin: 0 0 4px; font-size: 24px; letter-spacing: 0; }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(260px, 2fr) repeat(4, minmax(138px, 1fr)) auto auto;
      gap: 10px;
      align-items: end;
      margin-top: 16px;
    }}
    label {{ display: grid; gap: 5px; color: var(--muted); font-size: 12px; font-weight: 800; }}
    select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      font: inherit;
    }}
    button {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 9px 12px;
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{ border-color: #aab5ad; }}
    button.primary {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .tabs {{ display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }}
    .tab.active {{ background: var(--ink); color: #fff; border-color: var(--ink); }}
    main {{ max-width: 1560px; margin: 0 auto; padding: 20px 22px 42px; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .stat {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; box-shadow: var(--shadow); }}
    .stat b {{ display: block; font-size: 18px; }}
    .source, .totals {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
      box-shadow: var(--shadow);
    }}
    .source-title, .totals-title {{ font-weight: 900; margin-bottom: 7px; }}
    .chips {{ display: flex; gap: 7px; flex-wrap: wrap; margin-top: 9px; }}
    .chip {{ border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; background: #fafbf9; color: var(--muted); font-size: 12px; }}
    .grid {{ display: grid; gap: 12px; align-items: stretch; }}
    .grid.model, .grid.constraint {{ grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); }}
    .grid.temp {{ grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
    .grid.total {{ grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; box-shadow: var(--shadow); min-height: 180px; }}
    .card.preferred {{ border-color: var(--gold); box-shadow: 0 0 0 2px rgba(153,106,0,.16), var(--shadow); }}
    .card-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: start; margin-bottom: 10px; }}
    .card-title {{ font-weight: 900; }}
    .meta {{ color: var(--muted); font-size: 12px; }}
    .query {{ border-left: 4px solid var(--accent); background: var(--accent-weak); padding: 10px 12px; margin: 10px 0; border-radius: 0 6px 6px 0; white-space: pre-wrap; }}
    details {{ margin-top: 9px; border-top: 1px solid var(--line); padding-top: 8px; }}
    summary {{ cursor: pointer; color: var(--muted); font-weight: 800; }}
    .explanation {{ white-space: pre-wrap; color: #34414f; margin-top: 8px; }}
    .actions {{ display: flex; gap: 6px; margin-top: 12px; flex-wrap: wrap; }}
    .actions button {{ padding: 6px 9px; font-size: 12px; }}
    .actions button.selected[data-value="ok"] {{ background: #e5f3e9; border-color: var(--ok); color: var(--ok); }}
    .actions button.selected[data-value="issue"] {{ background: #fff5db; border-color: var(--gold); color: var(--gold); }}
    .actions button.selected[data-value="bad"] {{ background: #fde8e8; border-color: var(--bad); color: var(--bad); }}
    .prefer.selected {{ background: var(--gold-weak); border-color: var(--gold); color: var(--gold); font-weight: 900; }}
    .missing {{ color: var(--muted); font-style: italic; }}
    .total-row-title {{ font-weight: 900; margin: 18px 0 9px; font-size: 16px; }}
    .totals-table {{ display: grid; gap: 6px; }}
    .totals-row {{ display: grid; grid-template-columns: minmax(220px, 1fr) 80px; gap: 10px; border-top: 1px solid var(--line); padding-top: 6px; }}
    @media (max-width: 1050px) {{ .toolbar {{ grid-template-columns: 1fr 1fr; }} .stats {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 560px) {{ .toolbar, .stats {{ grid-template-columns: 1fr; }} .wrap, main {{ padding-left: 14px; padding-right: 14px; }} }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>{html.escape(title)}</h1>
      <div class="sub" id="subtitle"></div>
      <div class="toolbar">
        <label>Source Caption <select id="caseSelect"></select></label>
        <label>EQ Type <select id="typeSelect"></select></label>
        <label>Constraint <select id="constraintSelect"></select></label>
        <label>Model <select id="modelSelect"></select></label>
        <label>Temperature <select id="tempSelect"></select></label>
        <button id="resetFilters">초기화</button>
        <button class="primary" id="exportState">검수 JSON 내보내기</button>
      </div>
      <div class="tabs">
        <button class="tab active" data-tab="model">모델별 비교</button>
        <button class="tab" data-tab="temp">Temperature 비교</button>
        <button class="tab" data-tab="constraint">Constraint 비교</button>
        <button class="tab" data-tab="total">총 비교</button>
      </div>
    </div>
  </header>
  <main>
    <section class="stats" id="stats"></section>
    <section class="source" id="sourceBox"></section>
    <section class="totals" id="preferenceTotals"></section>
    <section id="content"></section>
  </main>
  <script>
    const DATA = {data_json};
    const state = {{
      tab: "model",
      caseId: DATA.cases[0]?.audio_id || "",
      eqType: DATA.eq_types[0] || "",
      constraint: DATA.constraints[0]?.id || "",
      model: DATA.models[0] || "",
      temp: DATA.temps[0] || ""
    }};
    const reviewKey = "eqVerificationReview:" + DATA.root;
    const prefKey = "eqVerificationPreference:" + DATA.root;
    let reviews = JSON.parse(localStorage.getItem(reviewKey) || "{{}}");
    let preferences = JSON.parse(localStorage.getItem(prefKey) || "{{}}");
    const byKey = new Map(DATA.records.map(r => [`${{r.constraint_id}}|${{r.model}}|${{r.temp}}|${{r.eq_type}}|${{r.audio_id}}`, r]));

    function $(id) {{ return document.getElementById(id); }}
    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
    }}
    function caseLabel(item, idx) {{
      return `${{idx + 1}}. [${{item.dataset}}] ${{(item.source_caption || "").slice(0, 110)}}`;
    }}
    function constraintLabel(id) {{
      return DATA.constraints.find(c => c.id === id)?.label || id;
    }}
    function record(constraint, model, temp, eqType, caseId) {{
      return byKey.get(`${{constraint}}|${{model}}|${{temp}}|${{eqType}}|${{caseId}}`);
    }}
    function saveState() {{
      localStorage.setItem(reviewKey, JSON.stringify(reviews, null, 2));
      localStorage.setItem(prefKey, JSON.stringify(preferences, null, 2));
    }}
    function repeated(values) {{
      const counts = new Map();
      values.filter(Boolean).forEach(v => counts.set(v, (counts.get(v) || 0) + 1));
      return Array.from(counts.entries()).filter(([, n]) => n > 1).map(([v, n]) => `${{v}} x${{n}}`).join(", ") || "-";
    }}
    function comparisonKey() {{
      if (state.tab === "model") return `model|${{state.constraint}}|${{state.temp}}|${{state.eqType}}|${{state.caseId}}`;
      if (state.tab === "temp") return `temp|${{state.constraint}}|${{state.model}}|${{state.eqType}}|${{state.caseId}}`;
      if (state.tab === "constraint") return `constraint|${{state.model}}|${{state.temp}}|${{state.eqType}}|${{state.caseId}}`;
      return `total|${{state.eqType}}|${{state.caseId}}`;
    }}
    function fillSelects() {{
      $("caseSelect").innerHTML = DATA.cases.map((c, i) => `<option value="${{escapeHtml(c.audio_id)}}">${{escapeHtml(caseLabel(c, i))}}</option>`).join("");
      $("typeSelect").innerHTML = DATA.eq_types.map(t => `<option value="${{escapeHtml(t)}}">${{escapeHtml(t)}}</option>`).join("");
      $("constraintSelect").innerHTML = DATA.constraints.map(c => `<option value="${{escapeHtml(c.id)}}">${{escapeHtml(c.label)}} (${{escapeHtml(c.id)}})</option>`).join("");
      $("modelSelect").innerHTML = DATA.models.map(m => `<option value="${{escapeHtml(m)}}">${{escapeHtml(m)}}</option>`).join("");
      $("tempSelect").innerHTML = DATA.temps.map(t => `<option value="${{escapeHtml(t)}}">temp ${{escapeHtml(t)}}</option>`).join("");
      $("caseSelect").value = state.caseId;
      $("typeSelect").value = state.eqType;
      $("constraintSelect").value = state.constraint;
      $("modelSelect").value = state.model;
      $("tempSelect").value = state.temp;
    }}
    function bindControls() {{
      $("caseSelect").addEventListener("change", e => {{ state.caseId = e.target.value; render(); }});
      $("typeSelect").addEventListener("change", e => {{ state.eqType = e.target.value; render(); }});
      $("constraintSelect").addEventListener("change", e => {{ state.constraint = e.target.value; render(); }});
      $("modelSelect").addEventListener("change", e => {{ state.model = e.target.value; render(); }});
      $("tempSelect").addEventListener("change", e => {{ state.temp = e.target.value; render(); }});
      $("resetFilters").addEventListener("click", () => {{
        state.caseId = DATA.cases[0]?.audio_id || "";
        state.eqType = DATA.eq_types[0] || "";
        state.constraint = DATA.constraints[0]?.id || "";
        state.model = DATA.models[0] || "";
        state.temp = DATA.temps[0] || "";
        fillSelects();
        render();
      }});
      $("exportState").addEventListener("click", () => {{
        const payload = {{ root: DATA.root, exported_at: new Date().toISOString(), reviews, preferences }};
        const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "eq_verification_state.json";
        a.click();
        URL.revokeObjectURL(url);
      }});
      document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => {{
        state.tab = btn.dataset.tab;
        document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b === btn));
        render();
      }}));
    }}
    function card(rec, title, subtitle, prefScope) {{
      if (!rec) {{
        return `<article class="card"><div class="card-title">${{escapeHtml(title)}}</div><p class="missing">No generated query for this slot.</p></article>`;
      }}
      const review = reviews[rec.id] || "";
      const preferred = preferences[prefScope] === rec.id;
      const reviewButtons = [["ok", "통과"], ["issue", "확인"], ["bad", "문제"], ["clear", "해제"]].map(([value, label]) => {{
        const selected = review === value && value !== "clear" ? " selected" : "";
        return `<button data-kind="review" data-id="${{escapeHtml(rec.id)}}" data-value="${{value}}" class="${{selected}}">${{label}}</button>`;
      }}).join("");
      return `<article class="card${{preferred ? " preferred" : ""}}">
        <div class="card-head">
          <div>
            <div class="card-title">${{escapeHtml(title)}}</div>
            <div class="meta">${{escapeHtml(subtitle)}} · ${{escapeHtml(rec.path)}}:${{rec.line}}</div>
          </div>
          <div class="chips">
            <span class="chip">${{escapeHtml(rec.first_word || "-")}}</span>
            <span class="chip">${{escapeHtml(rec.first_3_words || "-")}}</span>
          </div>
        </div>
        <div class="query">${{escapeHtml(rec.generated_query)}}</div>
        <details>
          <summary>explanation</summary>
          <div class="explanation">${{escapeHtml(rec.explanation || "-")}}</div>
        </details>
        <div class="actions">
          <button data-kind="prefer" data-pref="${{escapeHtml(prefScope)}}" data-id="${{escapeHtml(rec.id)}}" class="prefer${{preferred ? " selected" : ""}}">선호</button>
          ${{reviewButtons}}
        </div>
      </article>`;
    }}
    function currentVisibleRecords() {{
      if (state.tab === "model") return DATA.models.map(m => record(state.constraint, m, state.temp, state.eqType, state.caseId)).filter(Boolean);
      if (state.tab === "temp") return DATA.temps.map(t => record(state.constraint, state.model, t, state.eqType, state.caseId)).filter(Boolean);
      if (state.tab === "constraint") return DATA.constraints.map(c => record(c.id, state.model, state.temp, state.eqType, state.caseId)).filter(Boolean);
      return DATA.records.filter(r => r.audio_id === state.caseId && r.eq_type === state.eqType);
    }}
    function preferenceTotalsForTab(tab) {{
      const totals = new Map();
      for (const [scope, recId] of Object.entries(preferences)) {{
        if (!scope.startsWith(tab + "|")) continue;
        const rec = DATA.records.find(r => r.id === recId);
        if (!rec) continue;
        let key = rec.model;
        if (tab === "temp") key = `temp ${{rec.temp}}`;
        if (tab === "constraint") key = rec.constraint_label;
        if (tab === "total") key = `${{rec.constraint_label}} · ${{rec.model}} · temp ${{rec.temp}}`;
        totals.set(key, (totals.get(key) || 0) + 1);
      }}
      return Array.from(totals.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    }}
    function renderStats() {{
      const visible = currentVisibleRecords();
      const reviewed = Object.keys(reviews).length;
      const preferred = Object.keys(preferences).length;
      $("stats").innerHTML = [
        ["Records", DATA.records.length],
        ["Visible", visible.length],
        ["Reviewed", reviewed],
        ["Preferences", preferred],
      ].map(([k, v]) => `<div class="stat"><span class="meta">${{k}}</span><b>${{v}}</b></div>`).join("");
    }}
    function renderSource() {{
      const item = DATA.cases.find(c => c.audio_id === state.caseId);
      if (!item) return;
      $("sourceBox").innerHTML = `<div class="source-title">${{escapeHtml(item.audio_id)}} · ${{escapeHtml(item.dataset)}} · ${{escapeHtml(state.eqType)}}</div>
        <div>${{escapeHtml(item.source_caption)}}</div>
        <div class="chips">
          <span class="chip">${{DATA.models.length}} models</span>
          <span class="chip">${{DATA.temps.length}} temperatures</span>
          <span class="chip">${{DATA.constraints.length}} constraint sessions</span>
          <span class="chip">1w repeats: ${{escapeHtml(repeated(currentVisibleRecords().map(r => r.first_word)))}}</span>
          <span class="chip">3w repeats: ${{escapeHtml(repeated(currentVisibleRecords().map(r => r.first_3_words)))}}</span>
        </div>`;
    }}
    function renderPreferenceTotals() {{
      const rows = preferenceTotalsForTab(state.tab);
      const label = {{
        model: "모델별 비교 선호 합계",
        temp: "Temperature 비교 선호 합계",
        constraint: "Constraint 비교 선호 합계",
        total: "총 비교 선호 합계"
      }}[state.tab];
      $("preferenceTotals").innerHTML = `<div class="totals-title">${{label}}</div>` +
        (rows.length ? `<div class="totals-table">${{rows.map(([k, v]) => `<div class="totals-row"><span>${{escapeHtml(k)}}</span><b>${{v}}</b></div>`).join("")}}</div>` : `<div class="meta">아직 선택된 선호 캡션이 없습니다.</div>`);
    }}
    function renderModelTab() {{
      const scope = comparisonKey();
      const cards = DATA.models.map(model => card(record(state.constraint, model, state.temp, state.eqType, state.caseId), model, `${{constraintLabel(state.constraint)}} · temp ${{state.temp}} · ${{state.eqType}}`, scope)).join("");
      $("content").innerHTML = `<div class="grid model">${{cards}}</div>`;
    }}
    function renderTempTab() {{
      const scope = comparisonKey();
      const cards = DATA.temps.map(temp => card(record(state.constraint, state.model, temp, state.eqType, state.caseId), `temp ${{temp}}`, `${{constraintLabel(state.constraint)}} · ${{state.model}} · ${{state.eqType}}`, scope)).join("");
      $("content").innerHTML = `<div class="grid temp">${{cards}}</div>`;
    }}
    function renderConstraintTab() {{
      const scope = comparisonKey();
      const cards = DATA.constraints.map(c => card(record(c.id, state.model, state.temp, state.eqType, state.caseId), c.label, `${{c.id}} · ${{state.model}} · temp ${{state.temp}} · ${{state.eqType}}`, scope)).join("");
      $("content").innerHTML = `<div class="grid constraint">${{cards}}</div>`;
    }}
    function renderTotalTab() {{
      const scope = comparisonKey();
      const blocks = DATA.constraints.map(c => DATA.models.map(model => {{
        const cards = DATA.temps.map(temp => card(record(c.id, model, temp, state.eqType, state.caseId), `temp ${{temp}}`, `${{c.label}} · ${{model}} · ${{state.eqType}}`, scope)).join("");
        return `<div class="total-row-title">${{escapeHtml(c.label)}} · ${{escapeHtml(model)}}</div><div class="grid total">${{cards}}</div>`;
      }}).join("")).join("");
      $("content").innerHTML = blocks;
    }}
    function bindActionButtons() {{
      document.querySelectorAll("[data-kind='review']").forEach(btn => btn.addEventListener("click", () => {{
        const id = btn.dataset.id;
        const value = btn.dataset.value;
        if (value === "clear") delete reviews[id];
        else reviews[id] = value;
        saveState();
        render();
      }}));
      document.querySelectorAll("[data-kind='prefer']").forEach(btn => btn.addEventListener("click", () => {{
        const scope = btn.dataset.pref;
        const id = btn.dataset.id;
        if (preferences[scope] === id) delete preferences[scope];
        else preferences[scope] = id;
        saveState();
        render();
      }}));
    }}
    function render() {{
      $("subtitle").innerHTML = `Models: ${{DATA.models.join(" · ")}} · Constraints: ${{DATA.constraints.map(c => c.label + " (" + c.id + ")").join(" · ")}} · Source: ${{escapeHtml(DATA.root)}}`;
      renderStats();
      renderSource();
      renderPreferenceTotals();
      if (state.tab === "model") renderModelTab();
      if (state.tab === "temp") renderTempTab();
      if (state.tab === "constraint") renderConstraintTab();
      if (state.tab === "total") renderTotalTab();
      bindActionButtons();
    }}
    fillSelects();
    bindControls();
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static EQ verification site.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/eq_recall_cases_temps/topp_0_9"),
        help="Root directory containing timestamp session directories.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=["20260515_095050=yes constraint", "20260516_095649=no constraint"],
        help="Session mappings in SESSION=LABEL form.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4.1-2025-04-14", "gpt-5.1-2025-11-13"],
        help="Model directories to include from each session.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to ROOT/verification_site/index.html.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_path = args.out.resolve() if args.out else root / "verification_site" / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_records(root, [parse_session(s) for s in args.sessions], args.models)
    output_path.write_text(render_html(data), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
