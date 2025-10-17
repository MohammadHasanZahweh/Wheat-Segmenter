# Wheat-Segmenter — Local Run Guide

This is a concise, copy‑ready README for your local version extracted from Colab. It explains setup, data layout, and exact commands to run on Windows/VS Code without Google Drive.

## What This Repo Contains

- `wheat_segmenter.py`: Single‑file runner/dataset loader extracted from your Colab notebook.
- `requirements.txt`: Python dependencies.
- `.vscode/settings.json`: Points VS Code to the local venv (optional, already added).

## Prerequisites

- Python 3.10 or 3.12 (Windows). 3.13 is not recommended for rasterio/PyTorch wheels yet.
- VS Code with Python extension (optional, but recommended).

## Environment Setup (Windows PowerShell)

```powershell
# From the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# If PyTorch fails to resolve, install CPU wheels explicitly
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Data Layout

Pass `--root` pointing to the folder that directly contains `data/` and `label/`:

```
<ROOT>/data/<YEAR>/<REGION>/<MONTH>/<TILE_ID>.tif
<ROOT>/label/<YEAR>/<REGION>/<TILE_ID>.tif
```

Example folder you showed:

```
C:\Users\Administrator\Desktop\preprocessed_data
  ├─ data\
  └─ label\
```

## Quickstart Commands

Activate the venv each new terminal session:

```powershell
.\.venv\Scripts\Activate.ps1
```

1) Summary check (counts tiles, validates shapes on a few files)

```powershell
python wheat_segmenter.py --root "C:\Users\Administrator\Desktop\preprocessed_data" --year 2020 --summary
```

2) Demo loader (picks a small subset and prints tensor shapes)

```powershell
python wheat_segmenter.py --root "C:\Users\Administrator\Desktop\preprocessed_data" --year 2020 --batch 8 --subset_k 32
```

Optional flags:

- `--regions 0 1 2 3 4`  (limit to specific regions)
- `--months 11 12 1 2 3 4 5 6 7`  (choose month sequence)

## VS Code Debug (F5)

Create `.vscode/launch.json` with this config:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Wheat demo",
      "type": "python",
      "request": "launch",
      "program": "wheat_segmenter.py",
      "console": "integratedTerminal",
      "args": [
        "--root", "C:\\Users\\Administrator\\Desktop\\preprocessed_data",
        "--year", "2020",
        "--batch", "8",
        "--subset_k", "32"
      ]
    },
    {
      "name": "Wheat summary",
      "type": "python",
      "request": "launch",
      "program": "wheat_segmenter.py",
      "console": "integratedTerminal",
      "args": [
        "--root", "C:\\Users\\Administrator\\Desktop\\preprocessed_data",
        "--year", "2020",
        "--summary"
      ]
    }
  ]
}
```

Ensure the interpreter is `.venv` (VS Code status bar → select interpreter) and reload the window if IntelliSense shows missing imports.

## Editing Defaults (Optional)

To avoid passing CLI args each time, edit at the top of `wheat_segmenter.py`:

```
ROOT = r"C:\Users\Administrator\Desktop\preprocessed_data"
YEAR = "2020"
REGIONS = None
MONTHS = (11,12,1,2,3,4,5,6,7)
```

Then you can run:

```powershell
python wheat_segmenter.py --summary
python wheat_segmenter.py --batch 8 --subset_k 32
```

## Troubleshooting

- FileNotFoundError: Make sure `--root` directly contains `data\` and `label\`, and `--year` exists under both.
- Missing imports in VS Code: Select the `.venv` interpreter and reload the window.
- Rasterio install errors on 3.13: Prefer Python 3.10/3.12.
- No tiles selected: Reduce `--subset_k`, verify months exist for chosen regions, or adjust `MONTHS`.

