#!/usr/bin/env bash
# Build docs/pyOCEAN_manual.pdf from the MkDocs manual (docs/manual/).
#
# Steps: (1) derive a print configuration from mkdocs.yml (mkdocs-print-site-plugin, flat URLs, Mermaid
# fences rendered by javascripts/mermaid-print.js outside Material's shadow DOM so stylesheets/print.css can
# fit each diagram on one page); (2) build the single print page; (3) serve it on localhost and print it
# with headless Chrome, giving MathJax and Mermaid (both from jsDelivr, so network is needed) time to render.
#
# Needs the docs venv (python -m venv .venv-docs && .venv-docs/bin/pip install -r docs/requirements-docs.txt)
# and Google Chrome.  Usage:  scripts/build_manual_pdf.sh [output.pdf]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/docs/pyOCEAN_manual.pdf}"
MKDOCS="${MKDOCS:-$ROOT/.venv-docs/bin/mkdocs}"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PORT="${PORT:-8767}"
WORK="$(mktemp -d)"
trap 'kill "${SERVER_PID:-0}" 2>/dev/null || true; rm -rf "$WORK"' EXIT

python3 - "$ROOT/mkdocs.yml" "$WORK/mkdocs_pdf.yml" "$ROOT/docs/manual" <<'EOF'
import sys
src, dst, docs = sys.argv[1:]
t = open(src).read()
t = t.replace("docs_dir: docs/manual", f"docs_dir: {docs}\nuse_directory_urls: false")
t = t.replace("class: mermaid\n", "class: mermaid-print\n")
t = t.replace("extra_javascript:\n", "extra_javascript:\n"
              "  - https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js\n"
              "  - javascripts/mermaid-print.js\n", 1)
t = t.split("\nvalidation:")[0]          # link checks are done by the normal build
t += """
plugins:
  - search
  - print-site:
      add_to_navigation: false
      print_page_title: "pyOCEAN Model Manual"
      add_print_site_banner: false
      add_table_of_contents: true
      toc_title: "Contents"
      toc_depth: 2
      add_full_urls: false
      enumerate_headings: true
      enumerate_headings_depth: 2
      enumerate_figures: true
      add_cover_page: true
"""
open(dst, "w").write(t)
EOF

"$MKDOCS" build -q -f "$WORK/mkdocs_pdf.yml" -d "$WORK/site"
(cd "$WORK/site" && exec python3 -m http.server "$PORT" >/dev/null 2>&1) &
SERVER_PID=$!
disown "$SERVER_PID"
sleep 1
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --run-all-compositor-stages-before-draw \
    --virtual-time-budget=120000 --user-data-dir="$WORK/chrome" \
    --print-to-pdf="$OUT" "http://127.0.0.1:$PORT/print_page.html" 2>/dev/null
command -v pdfinfo >/dev/null && pdfinfo "$OUT" | grep Pages
echo "wrote $OUT"
