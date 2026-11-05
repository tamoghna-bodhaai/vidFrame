"""Source-grounded multimodal study-note assembly and safe LaTex PDF export."""
from __future__ import annotations

import json
import html
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

from .diagrams import collect_diagrams
from .utils import fmt_ts, now_iso

_MATH = re.compile(r"^[A-Za-z0-9\\\\^_{}()+\-*/=<>.,:;| \[\]]+$")
_MATH_COMMANDS = {
    "alpha", "beta", "cdot", "delta", "Delta", "epsilon", "eta", "exp", "frac", "gamma",
    "Gamma", "infty", "lambda", "Lambda", "left", "ln", "log", "mu", "nu", "omega", "Omega",
    "phi", "pi", "Pi", "pm", "right", "rho", "sigma", "Sigma", "sin", "sqrt", "sum", "tau",
    "theta", "Theta", "times", "xi", "Xi",
}


_DIAGRAM_STOPWORDS = {
    "a", "an", "and", "at", "board", "chart", "class", "diagram", "frame", "in", "of",
    "on", "other", "presentation", "screen", "slide", "the", "this", "to", "with", "whiteboard",
}


def _diagram_terms(diagram: dict) -> set[str]:
    """Stable content terms used only to collapse repeated visual states."""
    text = " ".join(
        str(value)
        for value in [
            diagram.get("title", ""),
            *list(diagram.get("nodes") or [])[:12],
            *list(diagram.get("labels") or [])[:16],
            *list(diagram.get("values") or [])[:8],
        ]
    ).lower()
    return {
        term for term in re.findall(r"[a-z0-9α-ωβγλμνρτφχψω]{2,}", text)
        if term not in _DIAGRAM_STOPWORDS
    }


def diagram_selection_candidates(diagrams: list[dict], limit: int = 80) -> list[dict]:
    """Collapse repeated whiteboard states before the compiler decides what to illustrate.

    This is intentionally a *candidate* list, not the final diagram list.  It keeps the
    strongest source reference for each repeated concept so the text compiler sees the
    teaching material rather than hundreds of near-identical frames.
    """
    groups: list[tuple[set[str], dict]] = []
    for diagram in diagrams:
        if float(diagram.get("confidence") or 0) < 0.45:
            continue
        terms = _diagram_terms(diagram)
        # A detected box with no readable structure is poor material for a regenerated
        # teaching graphic. Keep it out of the paid-generation decision.
        if len(terms) < 2:
            continue
        best: int | None = None
        best_score = 0.0
        for position, (existing_terms, _existing) in enumerate(groups):
            overlap = len(terms & existing_terms) / max(1, len(terms | existing_terms))
            if overlap > best_score:
                best, best_score = position, overlap
        confidence = float(diagram.get("confidence") or 0)
        if best is not None and best_score >= 0.55:
            old_terms, old = groups[best]
            if confidence > float(old.get("confidence") or 0):
                groups[best] = (old_terms | terms, dict(diagram))
            else:
                groups[best] = (old_terms | terms, old)
        else:
            groups.append((terms, dict(diagram)))
    # Prefer readable, structured source evidence; preserve chronology as a tiebreaker.
    ranked = sorted(
        (entry for _terms, entry in groups),
        key=lambda item: (
            -float(item.get("confidence") or 0),
            -(len(item.get("labels") or []) + len(item.get("nodes") or []) + len(item.get("values") or [])),
            float(item.get("time_seconds") or 0),
        ),
    )
    return ranked[:max(0, limit)]


def canonical_diagrams(diagrams: list[dict], limit: int) -> list[dict]:
    """Deterministic conservative fallback when the compiler is unavailable."""
    limit = max(0, limit)
    if not limit:
        return []
    chosen: list[dict] = []
    seen: set[str] = set()
    for diagram in diagram_selection_candidates(diagrams):
        key = " ".join(sorted(_diagram_terms(diagram)))
        if not key or key in seen:
            continue
        seen.add(key)
        chosen.append(dict(diagram))
        if len(chosen) >= limit:
            break
    return chosen

def approved_latex(value: str, confidence: float) -> str:
    value = (value or "").strip()
    if confidence < .65 or len(value) > 500 or not _MATH.fullmatch(value):
        return ""
    # Never pass an unbalanced or unknown command to pdflatex. A visual model can
    # emit a plausible-looking but truncated equation; source text is safer there.
    depth = 0
    for char in value:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return ""
    if depth:
        return ""
    commands = re.findall(r"\\([A-Za-z]+)", value)
    if any(command not in _MATH_COMMANDS for command in commands):
        return ""
    # TeX accepts a dangling '_' far enough to corrupt the rest of the document.
    # Restrict superscripts/subscripts to one token or a balanced braced group.
    if re.search(r"[_^](?![A-Za-z0-9{])", value):
        return ""
    # frac requires both braced arguments. This catches the common truncated
    # '\\frac{dN' failure mode before TeX sees it.
    if re.search(r"\\frac(?!\s*\{[^{}]*\}\s*\{[^{}]*\})", value):
        return ""
    return value

def build_evidence(frames: list[dict], observations: dict[int, dict], transcript: dict) -> tuple[list[dict], list[dict]]:
    evidence, equations = [], []
    segments = transcript.get("segments") or []
    for frame in frames:
        obs = observations.get(frame["index"], {})
        if obs.get("status") not in {"ok", "partial"}: continue
        time = float(frame.get("time", 0))
        spoken = [s for s in segments if float(s.get("start", 0)) <= time + 2 and float(s.get("end", 0)) >= time - 2]
        evidence.append({"timestamp": time, "frame_index": frame["index"], "scene": obs.get("scene") or obs.get("summary"), "visual": obs.get("description") or obs.get("summary"), "transcript": spoken, "source_frame": frame.get("file")})
        for item in obs.get("equations") or []:
            raw = str(item.get("raw", "")).strip()
            if raw:
                confidence = float(item.get("confidence", 0))
                equations.append({"raw": raw, "latex": approved_latex(str(item.get("latex", "")), confidence), "confidence": confidence, "time": fmt_ts(time), "frame_index": frame["index"], "explanation": obs.get("description") or ""})
    return evidence, equations

def evidence_chunks(transcript: dict, evidence: list[dict], max_chars: int = 12000) -> list[dict]:
    """Partition all speech once and include teaching visuals in their time window."""
    segments = sorted(transcript.get("segments") or [], key=lambda s: float(s.get("start") or 0))
    chunks, current, size = [], [], 0
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        if current and size + len(text) > max_chars:
            chunks.append(current); current = []; size = 0
        current.append(segment); size += len(text)
    if current:
        chunks.append(current)
    if not chunks and evidence:
        # A silent video still has visual evidence, partitioned into bounded batches.
        return [{"start": part[0]["timestamp"], "end": part[-1]["timestamp"] + 0.001, "segments": [], "visuals": part} for part in (evidence[i:i+50] for i in range(0, len(evidence), 50))]
    result = []
    for i, part in enumerate(chunks):
        start = 0.0 if i == 0 else float(part[0].get("start") or 0)
        end = float(chunks[i+1][0].get("start") or 0) if i+1 < len(chunks) else float("inf")
        result.append({"start": start, "end": end, "segments": part, "visuals": [e for e in evidence if start <= e["timestamp"] < end]})
    return result


def compile_study(title: str, frames: list[dict], observations: dict[int, dict], transcript: dict,
                  meta: dict, selected_diagrams: list[dict] | None = None,
                  sections: list[dict] | None = None, compilation_warning: str = "") -> dict:
    evidence, raw_equations = build_evidence(frames, observations, transcript)
    diagram_evidence = collect_diagrams(frames, observations)
    diagrams = selected_diagrams if selected_diagrams is not None else diagram_selection_candidates(diagram_evidence, limit=len(diagram_evidence))
    # Repeated frames commonly contain the same formula. Preserve the first occurrence.
    equations, seen = [], set()
    for equation in raw_equations:
        key = re.sub(r"\s+", "", equation.get("latex") or equation["raw"]).lower()
        if key and key not in seen:
            seen.add(key); equations.append(equation)
    chapters = sections or []
    if not chapters:
        # Clearly marked source fallback: never claim an unsynthesized transcript is study notes.
        chapters = [{"title": "Source material (study-note synthesis unavailable)", "start_seconds": 0,
                     "subtopics": [{"title": "Transcript", "explanation": [str(s.get("text") or "") for s in transcript.get("segments") or [] if s.get("text")]}]}]
        compilation_warning = compilation_warning or "Study-note synthesis was unavailable; source transcript is shown instead."
    chapters.sort(key=lambda c: float(c.get("start_seconds") or 0))
    def seconds(value: str) -> float:
        try:
            result = 0.0
            for part in str(value).split(":"): result = result * 60 + float(part)
            return result
        except (ValueError, TypeError): return 0.0
    for chapter in chapters:
        chapter.setdefault("subtopics", [])
        chapter["equations"] = []; chapter["diagrams"] = []
        for sub in chapter["subtopics"]:
            sub["equations"] = []; sub["diagrams"] = []
    def place(item: dict, kind: str, when: float) -> None:
        index = max((i for i, c in enumerate(chapters) if float(c.get("start_seconds") or 0) <= when), default=0)
        chapter = chapters[index]
        subs = chapter.get("subtopics") or []
        eligible = [sub for sub in subs if float(sub.get("start_seconds", chapter.get("start_seconds", 0)) or 0) <= when]
        if eligible:
            max(eligible, key=lambda sub: float(sub.get("start_seconds", chapter.get("start_seconds", 0)) or 0))[kind].append(item)
        else:
            chapter[kind].append(item)
    # Raw OCR equations are evidence for synthesis, not automatically publishable formulas.
    for diagram in diagrams: place(diagram, "diagrams", float(diagram.get("time_seconds") or seconds(diagram.get("time", "0"))))
    return {"schema_version": 3, "kind": "multimodal-study-notes", "title": title, "summary": "", "language": transcript.get("language", "auto"), "generated_at": now_iso(), "meta": meta, "chapters": chapters, "equations": equations, "diagrams": diagrams, "evidence": evidence, "timeline": [], "compilation_warning": compilation_warning, "stats": {"diagrams": len(diagrams), "diagram_evidence": len(diagram_evidence), "equations": len(equations)}, "provider": meta.get("provider"), "transcript": {"available": bool(transcript.get("segments")), "segments": len(transcript.get("segments") or [])}}


def study_markdown(note: dict) -> str:
    out = [f"# {note['title']}", ""]
    if note.get("compilation_warning"): out += [f"**Compilation warning:** {note['compilation_warning']}", ""]
    for chapter in note.get("chapters", []):
        out += [f"## {chapter['title']}", ""]
        def assets(container: dict) -> None:
            for e in container.get("equations", []): out.extend([f"`{e.get('latex') or e.get('raw')}`", ""])
            for d in container.get("diagrams", []):
                out.extend([f"**{d.get('title') or 'Diagram'}**", "", d.get("description") or "", ""])
                image = d.get("generated_image") if str(d.get("generated_image") or "").endswith(".svg") else None
                if image and note.get("meta", {}).get("job_id"):
                    out.extend([f"![Source diagram](/api/jobs/{note['meta']['job_id']}/{image})", ""])
        assets(chapter)
        for sub in chapter.get("subtopics", []):
            out += [f"### {sub.get('title', 'Explanation')}", ""]
            out += [f"{text}\n" for text in sub.get("explanation", [])]
            assets(sub)
    return "\n".join(out)

def _tex(text: str) -> str:
    substitutions = str.maketrans({
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        "−": "-", "–": "-", "—": "-", "→": "->", "×": "x", "≤": "<=", "≥": ">=", "≈": "approximately ",
        "Δ": "Delta ", "δ": "delta ", "λ": "lambda ", "α": "alpha ", "β": "beta ", "γ": "gamma ", "τ": "tau ", "ν": "nu ", "μ": "mu ",
    })
    plain = unicodedata.normalize("NFKD", str(text).translate(substitutions)).encode("ascii", "ignore").decode("ascii")
    return plain.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("$", r"\$").replace("#", r"\#").replace("_", r"\_").replace("{", r"\{").replace("}", r"\}").replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")

def _html_text(value: str) -> str:
    """Escape prose while preserving explicitly marked KaTeX math."""
    value = str(value).replace("\x08", r"\b").replace("\x0c", r"\f")
    parts = re.split(r"(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\])", str(value))
    out = []
    for part in parts:
        if part.startswith("$$") and part.endswith("$$"):
            out.append(f'<div class="math display" data-tex="{html.escape(part[2:-2], quote=True)}"></div>')
        elif part.startswith(r"\[") and part.endswith(r"\]"):
            out.append(f'<div class="math display" data-tex="{html.escape(part[2:-2], quote=True)}"></div>')
        elif (part.startswith("$") and part.endswith("$")) or (part.startswith(r"\(") and part.endswith(r"\)")):
            out.append(f'<span class="math" data-tex="{html.escape(part[1:-1] if part.startswith("$") else part[2:-2], quote=True)}"></span>')
        else:
            out.append(html.escape(part))
    return "".join(out)


def export_pdf(note: dict, job_dir: Path, include_transcript: bool) -> tuple[Path, Path]:
    """Render structured notes in Chromium, using local KaTeX for marked math."""
    from playwright.sync_api import sync_playwright
    job_dir = job_dir.resolve()
    pdf_dir = job_dir / "pdf"; pdf_dir.mkdir(exist_ok=True)
    output = pdf_dir / "study-notes.pdf"; log = pdf_dir / "compile.log"
    katex_dir = Path(__file__).resolve().parent.parent / ".pdf-render/node_modules/katex/dist"
    if not (katex_dir / "katex.min.js").exists():
        raise RuntimeError("KaTeX is missing; run npm install --prefix .pdf-render")
    blocks = [f"<h1>{html.escape(note['title'])}</h1>"]
    if note.get("compilation_warning"):
        blocks.append(f'<p class="warning">{html.escape(note["compilation_warning"])}</p>')
    def assets(container: dict) -> None:
        for equation in container.get("equations", []):
            latex = equation.get("latex")
            blocks.append(f'<div class="equation">{_html_text("$$" + latex + "$$") if latex else html.escape(equation.get("raw", ""))}</div>')
        for diagram in container.get("diagrams", []):
            image = diagram.get("generated_image")
            if not image or not str(image).endswith(".svg"):
                continue  # Only deterministic diagrams with verified connections enter the PDF.
            image_path = (job_dir / str(image)).resolve()
            if not image_path.is_file() or job_dir.resolve() not in image_path.parents:
                continue
            blocks.append(f'<figure><img src="{image_path.as_uri()}"><figcaption>{html.escape(diagram.get("title") or "Diagram")}</figcaption></figure>')
    for chapter in note.get("chapters", []):
        blocks.append(f'<h2>{html.escape(chapter["title"])}</h2>')
        assets(chapter)
        for sub in chapter.get("subtopics", []):
            blocks.append(f'<h3>{html.escape(sub.get("title") or "Explanation")}</h3>')
            for paragraph in sub.get("explanation", []):
                for part in re.split(r"\n\s*\n", str(paragraph)):
                    if part.strip(): blocks.append(f'<p>{_html_text(part.strip())}</p>')
            assets(sub)
    if include_transcript:
        blocks.append('<h2>Transcript appendix</h2>')
        for segment in note.get("_transcript_segments", []):
            blocks.append(f'<p><strong>{html.escape(fmt_ts(segment.get("start", 0)))}</strong> {_html_text(segment.get("text", ""))}</p>')
    css = (katex_dir / "katex.min.css").as_uri()
    js = (katex_dir / "katex.min.js").as_uri()
    page_html = f"""<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="{css}">
<style>@page{{size:A4;margin:19mm 18mm}}body{{font:11pt/1.5 Georgia,serif;color:#161616}}
h1{{font-size:20pt;margin:0 0 14mm}}h2{{font-size:15pt;margin:10mm 0 4mm;break-after:avoid}}
h3{{font-size:12pt;margin:7mm 0 2mm;break-after:avoid}}p{{margin:0 0 4mm;white-space:pre-line}}
.equation{{margin:6mm 0;text-align:center;break-inside:avoid}}.display{{margin:6mm 0}}
figure{{margin:7mm auto;text-align:center;break-inside:avoid}}figure img{{max-width:100%;max-height:115mm;object-fit:contain}}
figcaption{{font:9pt Arial,sans-serif;color:#555;margin-top:2mm}}.warning{{color:#8a3b10}}</style></head>
<body>{''.join(blocks)}<script src="{js}"></script><script>
for(const element of document.querySelectorAll('.math')){{try{{katex.render(element.dataset.tex,element,{{displayMode:element.classList.contains('display'),throwOnError:true}})}}catch(error){{element.textContent=element.dataset.tex;element.style.color='#a00'}}}}
</script></body></html>"""
    html_path = pdf_dir / "study-notes.html"; html_path.write_text(page_html, encoding="utf-8")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=["--no-sandbox", "--allow-file-access-from-files"])
        page = browser.new_page(viewport={"width": 1100, "height": 1400})
        page.goto(html_path.as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready")
        errors = page.locator('.math[style*="color"]').count()
        page.pdf(path=str(output), format="A4", print_background=True)
        browser.close()
    log.write_text(f"KaTeX render completed; math errors: {errors}\n", encoding="utf-8")
    if errors:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"KaTeX could not render {errors} formula(s)")
    return output, log
