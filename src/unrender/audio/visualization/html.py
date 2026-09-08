"""Dependency-free, audio-only HTML explorer."""

from __future__ import annotations

import html
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

from unrender.audio.visualization.bundle import (
    AnalysisBundle,
    json_safe_bundle,
    load_analysis_bundle,
)
from unrender.audio.visualization.mapping import VisualMapping, load_visual_mapping


def _safe_json(value: Any) -> str:
    """Encode JSON so user strings cannot terminate an enclosing script element."""

    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _audio_items(
    paths: Sequence[str | Path],
    alternates: Sequence[str | Path],
    output: Path,
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for group, values in (("A", paths), ("B", alternates)):
        for index, raw_path in enumerate(values):
            path = Path(raw_path).expanduser()
            try:
                relative = os.path.relpath(path.resolve(), output.parent.resolve())
            except OSError:
                relative = str(path)
            items.append(
                {
                    "group": group,
                    "label": f"{group}{index + 1}: {path.name}",
                    "url": quote(Path(relative).as_posix(), safe="/:"),
                }
            )
    return items


def render_standalone_html(
    bundle: AnalysisBundle | Mapping[str, Any] | str | Path,
    output_path: str | Path,
    mapping: VisualMapping | Mapping[str, Any] | str | Path | None = None,
    *,
    source_paths: Sequence[str | Path] = (),
    alternate_paths: Sequence[str | Path] = (),
    embed_json: bool = True,
    source_path: str | Path | None = None,
    alternate_path: str | Path | None = None,
) -> Path:
    """Write a self-contained HTML audio-analysis explorer.

    Embedded data works directly from ``file://``. A local JSON file picker is
    always present, so even ``embed_json=False`` does not require a web server.
    """

    loaded = load_analysis_bundle(bundle)
    visual = load_visual_mapping(mapping)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe_bundle(loaded) if embed_json else {}
    selected_sources = tuple(source_paths) + ((source_path,) if source_path is not None else ())
    selected_alternates = tuple(alternate_paths) + (
        (alternate_path,) if alternate_path is not None else ()
    )
    audio_items = _audio_items(selected_sources, selected_alternates, output)
    if not embed_json:
        sidecar = output.with_suffix(".analysis.json")
        sidecar.write_text(
            json.dumps(json_safe_bundle(loaded), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    options = "".join(
        f'<option value="{html.escape(item["url"], quote=True)}" '
        f'data-group="{item["group"]}">{html.escape(item["label"])}</option>'
        for item in audio_items
    )
    layer_controls = "".join(
        (
            '<label class="toggle"><input type="checkbox" class="layer-toggle" '
            f'data-layer="{html.escape(name, quote=True)}"'
            f"{' checked' if enabled else ''}> "
            f"{html.escape(name.replace('_', ' ').title())}</label>"
        )
        for name, enabled in visual.layers.items()
    )
    initial_audio = audio_items[0]["url"] if audio_items else ""
    colors = visual.colors
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audio Analysis Explorer</title>
<style>
:root {{
  color-scheme: dark;
  --bg: {html.escape(colors["background"])};
  --panel: {html.escape(colors["panel"])};
  --grid: {html.escape(colors["grid"])};
  --text: {html.escape(colors["text"])};
  --muted: {html.escape(colors["muted_text"])};
  --dx: {html.escape(colors["dialogue"])};
  --mx: {html.escape(colors["music"])};
  --fx: {html.escape(colors["fx"])};
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 20px; background: var(--bg); color: var(--text);
       font: 14px system-ui, sans-serif; }}
main {{ max-width: 1280px; margin: auto; }}
.panel {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 8px;
          padding: 14px; margin: 12px 0; }}
.controls {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }}
button, select, input {{ accent-color: var(--dx); }}
button, select {{ color: var(--text); background: var(--bg); border: 1px solid var(--grid);
                  border-radius: 4px; padding: 6px 10px; }}
button.active {{ outline: 2px solid var(--dx); }}
audio {{ width: 100%; margin-top: 10px; }}
#scrubber {{ width: 100%; }}
#atlas {{ width: 100%; height: 480px; background: var(--bg); border: 1px solid var(--grid); }}
.toggle {{ white-space: nowrap; color: var(--muted); }}
.time {{ min-width: 105px; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<main>
  <h1>Audio Analysis Explorer</h1>
  <section class="panel">
    <div class="controls">
      <label>Audio
        <select id="source-select" aria-label="Source or alternate audio">{options}</select>
      </label>
      <button type="button" id="a-button" aria-label="Select A source">A</button>
      <button type="button" id="b-button" aria-label="Select B alternate">B</button>
      <label>Local analysis JSON
        <input id="json-file" type="file" accept=".json,application/json">
      </label>
    </div>
    <audio id="audio-player" controls preload="metadata"
           src="{html.escape(initial_audio, quote=True)}"></audio>
    <div class="controls">
      <input id="scrubber" type="range" min="0" max="1000" value="0" aria-label="Audio scrubber">
      <span class="time" id="time-readout">0.00 / 0.00 s</span>
      <label>Zoom <input id="zoom" type="range" min="1" max="20" step="0.5" value="1"></label>
    </div>
  </section>
  <section class="panel controls" id="layer-controls">{layer_controls}</section>
  <canvas id="atlas" width="1200" height="480"
          aria-label="Interactive audio analysis timeline"></canvas>
</main>
<script type="application/json" id="analysis-data">{_safe_json(payload)}</script>
<script>
"use strict";
const palette = {_safe_json(colors)};
let analysis = JSON.parse(document.getElementById("analysis-data").textContent || "{{}}");
const audio = document.getElementById("audio-player");
const scrubber = document.getElementById("scrubber");
const zoom = document.getElementById("zoom");
const canvas = document.getElementById("atlas");
const context = canvas.getContext("2d");
const sourceSelect = document.getElementById("source-select");
const toggles = Array.from(document.querySelectorAll(".layer-toggle"));

function path(object, dotted) {{
  return dotted.split(".").reduce((value, key) => value && value[key], object);
}}
function feature(...names) {{
  for (const name of names) {{
    const value = path(analysis, name);
    if (value !== undefined && value !== null) return value;
  }}
  return [];
}}
function series(value) {{
  if (!Array.isArray(value)) return [];
  if (value.length && Array.isArray(value[0])) return value.map(row => Number(row[0]) || 0);
  return value.map(item => Number(item) || 0);
}}
function enabled(name) {{
  const toggle = toggles.find(item => item.dataset.layer === name);
  return !toggle || toggle.checked;
}}
function duration() {{
  return Number(analysis.duration_sec || (analysis.metadata && analysis.metadata.duration_sec)
    || audio.duration || 1);
}}
function drawLine(values, y0, height, color, viewStart, viewEnd) {{
  values = series(values);
  if (!values.length) return;
  let low = Math.min(...values), high = Math.max(...values);
  if (low === high) high = low + 1;
  context.strokeStyle = color; context.lineWidth = 2; context.beginPath();
  for (let x = 0; x < canvas.width; x++) {{
    const fraction = viewStart + (x / Math.max(1, canvas.width - 1)) * (viewEnd - viewStart);
    const index = Math.min(
      values.length - 1, Math.max(0, Math.round(fraction * (values.length - 1)))
    );
    const y = y0 + height - ((values[index] - low) / (high - low)) * height;
    if (x === 0) context.moveTo(x, y); else context.lineTo(x, y);
  }}
  context.stroke();
}}
function draw() {{
  context.fillStyle = palette.background; context.fillRect(0, 0, canvas.width, canvas.height);
  const scale = Number(zoom.value), center = (audio.currentTime || 0) / duration();
  const span = 1 / scale;
  const start = Math.max(0, Math.min(1 - span, center - span / 2)), end = start + span;
  const lanes = ["role_activity", "timbral_color", "stereo_space", "masking",
                 "loudness_dynamics", "events"];
  const laneHeight = canvas.height / lanes.length;
  lanes.forEach((name, index) => {{
    const y = index * laneHeight;
    context.fillStyle = index % 2 ? palette.panel : palette.background;
    context.fillRect(0, y, canvas.width, laneHeight);
    context.strokeStyle = palette.grid; context.strokeRect(0, y, canvas.width, laneHeight);
    context.fillStyle = palette.muted_text; context.fillText(name.replaceAll("_", " "), 8, y + 14);
  }});
  if (enabled("role_activity")) {{
    const roles = feature("features.role_activity", "role_activity");
    if (roles && !Array.isArray(roles)) {{
      drawLine(roles.dialogue || roles.dx, 4, laneHeight - 8, palette.dialogue, start, end);
      drawLine(roles.music || roles.mx, 4, laneHeight - 8, palette.music, start, end);
      drawLine(roles.fx || roles.sfx, 4, laneHeight - 8, palette.fx, start, end);
    }} else drawLine(roles, 4, laneHeight - 8, palette.dialogue, start, end);
  }}
  if (enabled("stereo_space"))
    drawLine(feature("features.stereo_space", "stereo_space"), laneHeight * 2 + 4,
             laneHeight - 8, palette.dialogue, start, end);
  if (enabled("masking"))
    drawLine(feature("features.masking", "masking"), laneHeight * 3 + 4,
             laneHeight - 8, palette.masking, start, end);
  if (enabled("loudness_dynamics")) {{
    drawLine(feature("features.loudness", "loudness"), laneHeight * 4 + 4,
             laneHeight - 8, palette.loudness, start, end);
    drawLine(feature("features.dynamics", "dynamics"), laneHeight * 4 + 4,
             laneHeight - 8, palette.dynamics, start, end);
  }}
  if (enabled("events")) {{
    const events = feature("events", "timeline.events") || [];
    for (const event of Array.isArray(events) ? events : []) {{
      const begin = Number(event.start_sec ?? event.start ?? 0) / duration();
      const finish = Number(event.end_sec ?? event.end ?? begin) / duration();
      if (finish < start || begin > end) continue;
      const x0 = (begin - start) / (end - start) * canvas.width;
      const x1 = (finish - start) / (end - start) * canvas.width;
      const role = String(event.role || event.type || "fx").toLowerCase();
      context.fillStyle = role.includes("dialog") || role.includes("speech") ? palette.dialogue
        : role.includes("music") ? palette.music : palette.fx;
      context.fillRect(x0, laneHeight * 5 + 20, Math.max(2, x1 - x0), laneHeight - 25);
    }}
  }}
  const playhead = ((audio.currentTime / duration()) - start) / (end - start) * canvas.width;
  context.strokeStyle = palette.playhead; context.beginPath(); context.moveTo(playhead, 0);
  context.lineTo(playhead, canvas.height); context.stroke();
}}
function selectGroup(group) {{
  const option = Array.from(sourceSelect.options).find(item => item.dataset.group === group);
  if (option) {{
    sourceSelect.value = option.value;
    sourceSelect.dispatchEvent(new Event("change"));
  }}
  document.getElementById("a-button").classList.toggle("active", group === "A");
  document.getElementById("b-button").classList.toggle("active", group === "B");
}}
sourceSelect.addEventListener("change", () => {{
  const time = audio.currentTime || 0; audio.src = sourceSelect.value;
  audio.addEventListener(
    "loadedmetadata",
    () => {{ audio.currentTime = Math.min(time, audio.duration || time); }},
    {{once: true}}
  );
}});
document.getElementById("a-button").addEventListener("click", () => selectGroup("A"));
document.getElementById("b-button").addEventListener("click", () => selectGroup("B"));
document.getElementById("json-file").addEventListener("change", async event => {{
  const file = event.target.files[0]; if (!file) return;
  analysis = JSON.parse(await file.text()); draw();
}});
scrubber.addEventListener("input", () => {{
  audio.currentTime = duration() * Number(scrubber.value) / 1000;
  draw();
}});
audio.addEventListener("timeupdate", () => {{
  scrubber.value = String(Math.round(1000 * audio.currentTime / duration()));
  document.getElementById("time-readout").textContent =
    `${{audio.currentTime.toFixed(2)}} / ${{duration().toFixed(2)}} s`;
  draw();
}});
zoom.addEventListener("input", draw);
toggles.forEach(toggle => toggle.addEventListener("change", draw));
window.addEventListener("resize", draw); draw();
</script>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    return output
