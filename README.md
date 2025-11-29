# 🌾 Wheat Segmentation from Multi-Temporal Satellite Imagery

**ML-based agricultural monitoring for Lebanon using Sentinel-2 satellite data**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.0-green.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

---

## 📋 Table of Contents

1. [Problem & Solution](#-problem--solution)
2. [Dataset](#-dataset)
4. [Quick Start (Docker)](#-quick-start-docker)
5. [Using the Application](#-using-the-application)
6. [Project Structure](#-project-structure)
7. [Pre-trained Models](#-pre-trained-models)

---

## 🌍 Problem & Solution
### The Challenge

**Agricultural monitoring in Lebanon** faces critical challenges:
- Manual field surveys are expensive, time-consuming, and dangerous in conflict zones
- Traditional remote sensing methods confuse wheat with other winter crops (barley, lentils)
- No accessible tools for non-experts to analyze satellite imagery

### Our Solution

**ML-powered web application** that enables:
- ✅ **Interactive map interface** - Draw region of interest with your mouse
- ✅ **No coding required** - Built for agricultural experts, not programmers
- ✅ **High accuracy** - F1 Score: 0.86, tested on 9-month satellite imagery

**How it works**: Analyzes 9 months × 13 spectral bands from Sentinel-2 satellite to classify each 10m pixel as wheat/non-wheat.

---
**Source**: Sentinel-2 Multi-Spectral Imagery
- **Coverage**: Lebanon (Bekaa Valley agricultural regions)
- **Temporal**: 9 months (Nov 2019 → Jul 2020) covering wheat growing season
- **Spectral**: 13 bands
| Month | Growth Stage | Characteristic |
|-------|-------------|----------------|
| Nov-Dec | Planting/Emergence | Low vegetation signal |
| Apr | Flowering | Peak vegetation signal |
| May-Jun | Grain Filling | Declining vegetation |
| Jul | Harvest | Harvested fields |

### Pre-trained Models Included

All models trained on 5% of 10% ofLebanon data, tested on remaining 95% of the 10%.:

| Model File | Threshold | F1 Score | IoU | Use Case |
|-----------|-----------|----------|-----|----------|
| `xgboost_5pct_threshold_0.5.sklearn.joblib` | 0.5 | ~0.86 | ~0.75 | **Balanced** (default) |
| `xgboost_5pct_threshold_0.6.sklearn.joblib` | 0.6 | ~0.84 | ~0.73 | Fewer false positives |
| `xgboost_5pct_threshold_0.65.sklearn.joblib` | 0.65 | ~0.83 | ~0.70 | **Conservative** (urban areas) |
| `xgboost_5pct_threshold_0.7.sklearn.joblib` | 0.7 | ~0.82 | ~0.65 | Very conservative |

- Use **threshold 0.5** for most agricultural regions
- Use **threshold 0.65** if you see false positives in cities/urban areas

---

## 🚀 Quick Start (Docker)

### Prerequisites

- **Docker Desktop** installed ([Download here](https://www.docker.com/products/docker-desktop))
- **4GB RAM** minimum (8GB recommended)
- **10GB disk space** for images and data

### Step 1: Clone Repository
```bash
git clone https://github.com/MohammadHasanZahweh/Wheat-Segmenter
cd Wheat-Segmenter
```
### Step 2: Launch Application

```bash
docker-compose -f docker-compose.gpu.yaml up --build
```

### Step 3: Access Web Interface

Open your browser and go to:
- 🌐 **Streamlit UI**: http://localhost:8501
- 📚 **API Endpoints**: http://localhost:8000/

---


### Interactive Workflow

1. **Open Streamlit UI** (http://localhost:8501)
2. **Select Model** from sidebar (default: `xgboost_5pct_threshold_0.5.sklearn.joblib`)
3. **Draw Region** on interactive map
   - Click polygon tool
   - Draw your area of interest
   - Must be within Lebanon coverage area
4. **Click "Run Inference"**
5. **View Results**
   - Wheat distribution map overlay
   - Statistics (% wheat coverage, total area)
   - Download options (GeoTIFF, PNG, statistics)

### Example Regions to Test

**Bekaa Valley (main wheat region)**:
- Coordinates: Lon 35.8-36.2, Lat 33.8-34.1
- High wheat density expected
- Coordinates: Lon 35.4-35.6, Lat 33.8-34.0
- Low wheat expected (use threshold 0.65 to reduce false positives)


```
Wheat-Segmenter/
│
├── 📂 server/                      # Backend (FastAPI + ML models)
│   ├── dataset/                    # Data loading utilities
│   ├── model/                      # Model wrappers (XGBoost, HistGB, RF)
│   ├── inference/                  # Inference pipeline
│   ├── meta/                      # Normalization statistics
│   └── server/                     # FastAPI application
│       └── app.py                  # API endpoints
│
├── 📂 apps/                        # Frontend (Streamlit web app)
│   └── streamlit_app/
│       ├── app.py                  # Main UI
│       ├── core/                   # Inference engine, API client
│       └── ui/                     # UI components
│
├── 📂 runs/wheat/                  # Pre-trained models ⭐
│   ├── xgboost_5pct_threshold_0.5.sklearn.joblib
│   ├── xgboost_5pct_threshold_0.6.sklearn.joblib
│   ├── xgboost_5pct_threshold_0.65.sklearn.joblib
│   └── xgboost_5pct_threshold_0.7.sklearn.joblib
│
│  
│
├── 🐳 docker-compose.gpu.yaml      # GPU deployment
└── 📄 README.md                    # This file
```

---


All models in `runs/wheat/` are ready to use immediately after cloning. No training required!

### Model Details

**Algorithm**: XGBoost (Gradient Boosted Trees)
- 400 estimators, max depth 8
- Trained on 117 features (9 months × 13 spectral bands)
- **Threshold**: Probability cutoff for classifying pixel as wheat
- **Higher threshold** → Fewer false positives, but may miss some wheat


### Changing Model in Streamlit

1. Click sidebar "Model Selection"
2. Choose from dropdown:
   - `xgboost_5pct_threshold_0.5.sklearn.joblib` (default)
   - `xgboost_5pct_threshold_0.6.sklearn.joblib`
   - `xgboost_5pct_threshold_0.65.sklearn.joblib`
   - `xgboost_5pct_threshold_0.7.sklearn.joblib`
3. Model automatically reloads

### Docker Environment Variables

Edit `docker-compose.gpu.yaml` to customize:

```yaml
environment:
  - DATA_DIR=/data          # Satellite imagery location
  - RUNS_DIR=/runs          # Model files location
  - RESULTS_DIR=/results    # Output save location
```

---


### "Model predicts 0% wheat everywhere"
**Cause**: Region is urban/non-agricultural or data issue  
**Fix**: 
1. Try different region (Bekaa Valley has most wheat)
2. Switch to model with lower threshold (0.5 instead of 0.7)

### "Inference is very slow"
**Cause**: Running on very large polygon  
**Fix**:
1. Draw smaller regions for testing

### "Too many noise pixels false positives in cities"
**Cause**: Threshold too low  
**Fix**: Switch to `xgboost_5pct_threshold_0.65.sklearn.joblib` or higher

---


### Architecture

**Backend (FastAPI)**:
- RESTful API for model inference
- Handles GeoTIFF processing
- Interactive Folium map
- Polygon drawing tools
- Sklearn XGBoost classifiers
- Z-score normalization (per-month global stats)

```
User draws polygon → API receives coordinates → 
Load satellite tiles → Normalize pixels → 
Model prediction → Generate mask → 
Return GeoTIFF + visualization
```

---


MIT License - See [LICENSE](LICENSE)

---


- **Sentinel-2** (ESA Copernicus) for satellite imagery
- **FastAPI** and **Streamlit** for framework

**Last Updated**: November 2025
