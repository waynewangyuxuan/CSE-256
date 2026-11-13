#!/bin/sh
# Builds PA2_report.pdf: Part A from the training logs, Part B from the executed notebook.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
print_pdf() { "$CHROME" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$2" "file://$PWD/$1" 2>/dev/null; }

$PY report/build_report.py
print_pdf report/partA.html report/partA.pdf

$PY -m jupyter nbconvert --to html part3.ipynb --output-dir report --output partB_notebook >/dev/null 2>&1
$PY - <<'PYEOF'
path = "report/partB_notebook.html"
html = open(path, encoding="utf-8").read()
print_css = """<style>
@page { size: Letter; margin: 0.45in; }
@media print {
  .jp-InputPrompt, .jp-OutputPrompt { display: none !important; }
  pre, code { white-space: pre-wrap !important; overflow-wrap: anywhere; }
  .jp-OutputArea-output pre { font-size: 8.5pt !important; }
  .highlight pre { font-size: 8.5pt !important; }
  img { max-width: 100% !important; height: auto !important; }
  .jp-Cell { break-inside: auto; }
}
</style></head>"""
open(path, "w", encoding="utf-8").write(html.replace("</head>", print_css, 1))
PYEOF
print_pdf report/partB_notebook.html report/partB_notebook.pdf

$PY - <<'PYEOF'
from pypdf import PdfReader, PdfWriter
writer = PdfWriter()
for part in ("report/partA.pdf", "report/partB_notebook.pdf"):
    pages = PdfReader(part).pages
    print(part, len(pages), "pages")
    for page in pages:
        writer.add_page(page)
writer.write("PA2_report.pdf")
print("PA2_report.pdf", len(writer.pages), "pages")
PYEOF
