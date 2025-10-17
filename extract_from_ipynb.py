import json, re
from pathlib import Path
nb_path = Path('colab_notebook.ipynb')
out_path = Path('wheat_segmenter.py')
nb = json.loads(nb_path.read_text(encoding='utf-8'))
code_cells = [c for c in nb.get('cells', []) if c.get('cell_type')=='code']
lines_out = []
header = '''# Extracted from Colab notebook
# Colab magics and Drive mounts removed.
# Update DATA_ROOT below to point to your local data directory.
'''
lines_out.append(header)
lines_out.append("DATA_ROOT = r'./data'  # change if needed\n\n")
magic_re = re.compile(r'^\s*%')
colab_import_re = re.compile(r'\s*from\s+google\.colab\s+import\s+drive')
mount_re = re.compile(r'\s*drive\.mount\(.*\)')
cd_re = re.compile(r'^\s*!?cd\s+')
pip_re = re.compile(r'^\s*![pP]ip\s+|^\s*%pip\b')
path_replace_patterns = [
    (re.compile(r"/content/drive/MyDrive"), "DATA_ROOT"),
    (re.compile(r"/content"), "DATA_ROOT"),
]
for cell in code_cells:
    src = ''.join(cell.get('source', []))
    kept = []
    for ln in src.splitlines():
        if magic_re.match(ln):
            continue
        if colab_import_re.match(ln):
            continue
        if mount_re.match(ln):
            continue
        if cd_re.match(ln):
            continue
        if pip_re.match(ln):
            continue
        for rx, repl in path_replace_patterns:
            ln = rx.sub(repl, ln)
        kept.append(ln)
    if kept and any(s.strip() for s in kept):
        lines_out.append('\n# ==== Cell ====' + '\n')
        lines_out.append('\n'.join(kept) + '\n')
out_path.write_text(''.join(lines_out), encoding='utf-8')
print(f'Wrote {out_path} with extracted code cells: {len(code_cells)} cells processed')
