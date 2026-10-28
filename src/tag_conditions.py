"""Build a local HTML page for assigning a condition to each unsorted phone photo.

When photos were not shot one condition at a time, drop them
all in data/incoming/unsorted/, run this script, and open the page it writes.
Click one condition per photo (or "skip" to leave it out), then "Export CSV".
Save the download as data/incoming/assignments.csv and run
import_photos.py, which reads that manifest.

Thumbnails are written to data/incoming/thumbs/ with macOS `sips` so the page
loads fast and HEIC originals display in any browser.

Usage:
    python src/tag_conditions.py
    open data/incoming/tag_conditions.html
"""

import csv
import html
import json
import subprocess
from pathlib import Path

import config

INCOMING_DIR = config.DATA_DIR / "incoming"
UNSORTED_DIR = INCOMING_DIR / "unsorted"
THUMBS_DIR = INCOMING_DIR / "thumbs"
PAGE_PATH = INCOMING_DIR / "tag_conditions.html"
MANIFEST_PATH = INCOMING_DIR / "assignments.csv"
THUMB_MAX_EDGE = 700
RAW_EXTENSIONS = {".jpg", ".jpeg", ".heic", ".png"}
TAG_GROUPS = config.TAG_AXES


def make_thumbnail(source: Path) -> Path:
    thumb = THUMBS_DIR / f"{source.stem}.jpg"
    if not thumb.exists():
        subprocess.run(
            ["sips", "-s", "format", "jpeg", "-Z", str(THUMB_MAX_EDGE),
             str(source), "--out", str(thumb)],
            check=True, capture_output=True,
        )
    return thumb


def load_prefill() -> dict:
    """Seed the page with an existing assignments.csv so tags can be reviewed, not redone."""
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open(newline="") as handle:
        return {row.pop("file_name"): row for row in csv.DictReader(handle)}


def build_page(photos: list[Path]) -> str:
    prefill = json.dumps(load_prefill())
    # Two independent tag groups: one pick from each per photo.
    buttons = "".join(
        f'<div class="group">' + "".join(
            f'<button data-group="{group}" data-cond="{c}" onclick="pick(this)">{c}</button>'
            for c in options) + "</div>"
        for group, options in TAG_GROUPS.items()
    )
    cards = "\n".join(
        f'<div class="card" data-name="{html.escape(p.name)}">'
        f'<img src="thumbs/{html.escape(p.stem)}.jpg" loading="lazy">'
        f'<div class="name">{i + 1}. {html.escape(p.name)}</div>'
        f'<div class="buttons">{buttons}</div></div>'
        for i, p in enumerate(photos)
    )
    return f"""<!doctype html><meta charset="utf-8"><title>Tag conditions</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:16px;background:#111;color:#eee}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}}
.card{{background:#222;padding:8px;border-radius:8px}}
.card img{{width:100%;border-radius:4px}}
.group{{margin:3px 0}}
.name{{font-size:12px;color:#aaa;margin:6px 0}}
button{{margin:2px;padding:6px 10px;border:1px solid #555;background:#333;color:#eee;border-radius:4px;cursor:pointer}}
button.on{{background:#2a7;color:#000;font-weight:bold}}
#bar{{position:sticky;top:0;background:#111;padding:8px 0;border-bottom:1px solid #444;margin-bottom:12px}}
</style>
<div id="bar"><button onclick="exportCsv()">Export CSV</button> <span id="count"></span></div>
<div class="grid">{cards}</div>
<script>
const prefill = {prefill};
const picks = Object.assign({{}}, prefill, JSON.parse(localStorage.getItem('picks4') || '{{}}'));
function render() {{
  let done = 0;
  document.querySelectorAll('.card').forEach(card => {{
    const chosen = picks[card.dataset.name] || {{}};
    card.querySelectorAll('button').forEach(b => b.classList.toggle('on', chosen[b.dataset.group] === b.dataset.cond));
    if (Object.keys(chosen).length === 4) done++;
  }});
  const total = document.querySelectorAll('.card').length;
  document.getElementById('count').textContent = done + ' / ' + total + ' fully tagged';
}}
function pick(button) {{
  const name = button.closest('.card').dataset.name;
  picks[name] = picks[name] || {{}};
  picks[name][button.dataset.group] = button.dataset.cond;
  localStorage.setItem('picks4', JSON.stringify(picks));
  render();
}}
function exportCsv() {{
  let csv = 'file_name,boxes,blur,occlusion,color\\n';
  for (const [name, t] of Object.entries(picks)) if (Object.keys(t).length === 4) csv += name + ',' + t.boxes + ',' + t.blur + ',' + t.occlusion + ',' + t.color + '\\n';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {{type: 'text/csv'}}));
  a.download = 'assignments.csv'; a.click();
}}
render();
</script>"""


def main() -> None:
    THUMBS_DIR.mkdir(exist_ok=True)
    photos = sorted(p for p in UNSORTED_DIR.iterdir() if p.suffix.lower() in RAW_EXTENSIONS)
    for photo in photos:
        make_thumbnail(photo)
    PAGE_PATH.write_text(build_page(photos))
    print(f"{len(photos)} photos; open {PAGE_PATH}")


if __name__ == "__main__":
    main()
