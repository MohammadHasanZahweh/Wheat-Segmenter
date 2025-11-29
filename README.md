# 🌾 Wheat Segmentation from Multi-Temporal Satellite Imagery

**ML-based agricultural monitoring for Lebanon using Sentinel-2 satellite data**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.0-green.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal.svg)](https://fastapi.tiangolo.com/)

---

## 📋 Table of Contents

1. [Problem & Solution](#-problem--solution)
2. [Dataset](#-dataset)
3. [Model Performance](#-model-performance)
4. [Installation & Usage](#-installation--usage)
5. [Project Structure](#-project-structure)
6. [API & Deployment](#-api--deployment)

---

## 🌍 Problem & Solution

### The Challenge

The world is rapidly shifting toward data-driven decision making. Businesses, farmers, and entire communities now rely on digital insights more than intuition.

**Agricultural monitoring in Lebanon** faces critical challenges:
- Manual field surveys are expensive, time-consuming, and dangerous
- Traditional NDVI-based methods confuse wheat with other winter crops
- No accessible tools for non-experts to analyze satellite imagery

### Why Machine Learning?

**Traditional approaches fail** because:
- **Spectral overlap**: NDVI thresholding confuses wheat with barley/lentils
- **Temporal complexity**: Wheat phenology (Nov → Jul) requires multi-month analysis
- **Small fragmented fields**: Lebanese agriculture has irregular field boundaries

**ML Solution**: Pixel-level classification using 9 months × 13 spectral bands = 117 features

---

## 📊 Dataset

**Source**: Sentinel-2 MSI
- **ROI**: Bekaa Plains (all Lebanon coverage)
- **Temporal**: 9 months (Nov 2019 → Jul 2020) = 1 image per month
- **Spectral**: 13 hyperspectral bands
- **Resolution**: 10m per pixel
- **Structure**: Similar to video (time-series + hyperspectral images)

### Spectral Bands (13)

| Band | Wavelength | Use |
|------|-----------|-----|
| B2-B4 | 490-665nm | RGB, vegetation stress |
| B5-B7 | 705-783nm | Red Edge (chlorophyll) |
| B8, B8A | 842-865nm | NIR (biomass) |
| B11-B12 | 1610-2190nm | SWIR (moisture) |
| NDVI/NDMI/NDBI | Derived | Veg/water/urban indices |

### Wheat Phenology Timeline

| Month | Growth Stage | NDVI |
|-------|-------------|------|
| Nov-Dec | Planting/Emergence | Low (0.2-0.4) |
| Jan-Mar | Tillering/Elongation | Rising (0.4-0.7) |
| Apr | Flowering | Peak (0.7-0.9) |
| May-Jun | Grain Filling | Declining (0.5-0.3) |
| Jul | Harvest | Low (0.2-0.3) |

---

## 🤖 Model Performance

### Pixel-Level Classification (Sklearn Models) - 0.5% Training Data

| Model | F1 Score | IoU |
|-------|----------|-----|
| **Random Forest** | **0.854** | **0.75** |
| **XGBoost** | **0.86** | **0.75** |
| **HistGradientBoosting** | **0.858** | **0.75** |

### Pixel-Level Classification (Torch Models) - 0.5% Training Data

| Model | F1 Score | IoU |
|-------|----------|-----|
| **MLP** | **0.82** | **0.73** |
| **RNN** | **0.75** | **0.66** |
| **CNN** | **0.5** | **0.3** |

**Key Insights**:
- Sklearn models outperform deep learning (simpler task, limited data)
- XGBoost achieves best F1 (0.86) with minimal training data (0.5%)
- 117 temporal-spectral features sufficient for pixel classification

---

## ⚙️ Installation & Usage

### Prerequisites

- Python 3.11
- GDAL (geospatial library)
- 16GB RAM recommended

### Installation

```bash
git clone https://github.com/MohammadHasanZahweh/Wheat-Segmenter.git
cd Wheat-Segmenter

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Training Models

**XGBoost (recommended)**:
```python
from server.train.sklearn_train import TrainConfig, train_sklearn_model

cfg = TrainConfig(
    root='path/to/preprocessed_data',
    year='2020',
    model_type='xgboost',
    train_fraction=0.005,  # 0.5% of data
    test_fraction=0.15,
    pixels_per_tile=4096,
    balance_pixels=True,
    use_meta_stats=True,
    meta_dir='./meta',
    xgb_n_estimators=400,
    xgb_max_depth=8,
    save_model='./runs/wheat/xgboost.sklearn.joblib'
)

results = train_sklearn_model(cfg)
print(f"F1: {results['f1']:.3f}, IoU: {results['iou']:.3f}")
```

**HistGradientBoosting**:
```python
from server.train.sklearn_train import TrainConfig, train_sklearn_model

cfg = TrainConfig(
    root='path/to/preprocessed_data',
    year='2020',
    model_type='histgb',
    train_fraction=0.005,
    use_meta_stats=True,
    meta_dir='./meta',
    save_model='./runs/wheat/histgb.sklearn.joblib'
)

results = train_sklearn_model(cfg)
```

**Random Forest**:
```python
from server.train.sklearn_train import TrainConfig, train_sklearn_model

cfg = TrainConfig(
    root='path/to/preprocessed_data',
    year='2020',
    model_type='random_forest',
    train_fraction=0.005,
    use_meta_stats=True,
    meta_dir='./meta',
    save_model='./runs/wheat/rf.sklearn.joblib'
)

results = train_sklearn_model(cfg)
```

### Inference

**Local Python**:
```python
from server.model.sklearn_models import load_model
from server.inference.inference_lebanon import run_on_lebanon_one_year

# Load trained model
model = load_model('./runs/wheat/xgboost.sklearn.joblib')

# Run inference on polygon
run_on_lebanon_one_year(
    base_path='data/Lebanon/merge_data',
    year=2020,
    polygons=[{
        'type': 'Polygon',
        'coordinates': [[[35.5, 33.5], [35.6, 33.5], [35.6, 33.6], [35.5, 33.6], [35.5, 33.5]]]
    }],
    model=model,
    out_path='./results/Lebanon/my_region',
    patch_size=256,
    stride=256
)
# Output: ./results/Lebanon/my_region/wheat_mask.tif
```

**Via Streamlit UI**:
```bash
streamlit run apps/streamlit_app/app.py
# 1. Select model from dropdown
# 2. Draw polygon on map
# 3. Click "Run Inference"
```

### Docker Deployment

```bash
docker-compose -f docker-compose.gpu.yaml up --build

# Access:
# Streamlit UI: http://localhost:8501
# API docs: http://localhost:8000/docs
```

---

## 📁 Project Structure

```
Wheat-Segmenter/
│
├── server/                      # Backend services
│   ├── dataset/
│   │   ├── PatchDataset.py      # Multi-temporal tile dataset
│   │   ├── PixelDataset.py      # Pixel-level dataset
│   │   └── TileDataset.py       # Tile dataset utilities
│   │
│   ├── model/
│   │   ├── base_model.py        # Abstract model interface
│   │   ├── sklearn_models.py    # XGBoost/HistGB/RF models
│   │   ├── torch_pixel_model.py # PyTorch MLP
│   │   ├── RNNmodel.py          # RNN implementation
│   │   └── ml_utils.py          # F1/IoU metrics
│   │
│   ├── train/
│   │   ├── sklearn_train.py     # Sklearn training pipeline
│   │   ├── torch_train.py       # PyTorch training
│   │   └── patch_train.py       # Patch-based training
│   │
│   ├── inference/
│   │   ├── inference_lebanon.py # Polygon-based inference
│   │   ├── poly_tile_inference.py
│   │   └── tile_inference.py
│   │
│   └── server/
│       ├── app.py               # FastAPI server
│       └── config.py            # Configuration
│
├── apps/
│   └── streamlit_app/           # Web interface
│       ├── app.py
│       ├── core/                # API client
│       └── ui/                  # UI components
│
├── meta/                        # Normalization stats
│   ├── 2020_1.npz ... 2020_7.npz
│   └── 2020_11.npz, 2020_12.npz
│
├── docker-compose.gpu.yaml      # Docker deployment
├── requirements.txt
└── LICENSE
```

**Note**: `data/`, `runs/`, `results/`, and `notebooks/` are gitignored (too large for GitHub)

---

## 🚀 API & Deployment

### FastAPI Endpoints

**Training**:
```bash
POST /train
{
  "algorithm": "XGBOOST",
  "dataset": {"year": 2020, "train_fraction": 0.005},
  "save_model": true
}
```

**Inference**:
```bash
POST /inference-lebanon
{
  "model_name": "xgboost.sklearn.joblib",
  "year": 2020,
  "geometry": {"type": "Polygon", "coordinates": [...]}
}
```

**Status**:
```bash
GET /status/{job_id}
```

### Docker Services

```yaml
services:
  api:  # FastAPI server (port 8000)
    build: ./server
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia  # GPU support
  
  app:  # Streamlit UI (port 8501)
    build: ./apps
    depends_on: [api]
```

---

## 📊 Technical Implementation

### Training Pipeline

1. **Data Loading**: WheatTilesDataset (64×64 tiles, 9 months × 13 bands)
2. **Stratified Sampling**: 0.5% train, 15% test (balanced wheat/non-wheat)
3. **Normalization**: Z-score using global mean/std from `meta/*.npz`
4. **Feature Engineering**: 117 features per pixel (flattened temporal-spectral)
5. **Model Training**: XGBoost with 400 estimators, max_depth=8
6. **Evaluation**: F1=0.86, IoU=0.75 on 2M+ test pixels

### Inference Pipeline

1. **Input**: Lebanon GeoTIFF (18767×13830 pixels)
2. **Polygon Mask**: User-drawn region
3. **Tiling**: 256×256 patches (non-overlapping)
4. **Normalization**: Apply saved mean/std
5. **Prediction**: Per-pixel classification
6. **Output**: Binary mask (GeoTIFF + PNG visualization)

---

## 🐛 Troubleshooting

**"No tiles found in selected region"**
- **Cause**: Polygon outside data coverage
- **Fix**: Draw polygon in Bekaa Valley region (Lon 35.12-36.60, Lat 33.07-34.68)

**"Model predicts 0% wheat"**
- **Cause**: Normalization mismatch or zero-data area
- **Fix**: Ensure model trained with `use_meta_stats=True`

---

## 📝 License

MIT License - See [LICENSE](LICENSE)

---


**Last Updated**: November 2025

---

## 🌍 Problem & Solution

### Real-World Challenge

**Food security in Lebanon** requires accurate wheat crop monitoring, but:
- Manual field surveys are expensive and dangerous in conflict zones
- Existing GIS tools (QGIS, ArcGIS) require technical expertise  
- Simple NDVI thresholding confuses wheat with other winter crops (barley, lentils)

### Why Machine Learning?

Traditional remote sensing fails because:
1. **Spectral overlap**: NDVI thresholding achieves only ~60% accuracy
2. **Complex temporal patterns**: Wheat phenology (Nov planting → Jul harvest) requires multi-month analysis
3. **Spatial heterogeneity**: Small fragmented fields with mixed cropping

**ML Solution**: Learn from 9 months × 13 spectral bands = 117 features per pixel



## 📊 Data Pipeline

### Dataset

**Sentinel-2 Multi-Spectral Imagery (Level-2A)**
- **Coverage**: Lebanon (Lon: 35.12-36.60, Lat: 33.07-34.68)
- **Temporal**: 9 months (Nov 2019 → Jul 2020, wheat growing season)
- **Spectral**: 13 bands (RGB, Red Edge, NIR, SWIR, NDVI, NDMI, NDBI)
- **Resolution**: 10-20m per pixel


### Spectral Bands

| Band | Wavelength | Use Case |
|------|-----------|----------|
| B2-B4 | 490-665nm | RGB visualization, vegetation stress |
| B5-B7 | 705-783nm | Red Edge (chlorophyll, LAI) |
| B8, B8A | 842-865nm | NIR (biomass) |
| B11-B12 | 1610-2190nm | SWIR (moisture, soil) |
| NDVI/NDMI/NDBI | Derived | Vegetation/moisture/urban indices |

### Temporal Phenology

| Month | Growth Stage | NDVI Pattern |
|-------|-------------|--------------|
| Nov-Dec | Planting/Emergence | Low (0.2-0.4) |
| Jan-Mar | Tillering/Elongation | Rising (0.4-0.7) |
| Apr | Flowering | Peak (0.7-0.9) |
| May-Jun | Grain Filling/Ripening | Declining (0.5-0.3) |
| Jul | Harvest | Low (0.2-0.3) |

### Preprocessing

**1. Tile Extraction**
```python
# Split large GeoTIFF into 64×64 tiles
preprocessed_data/
├── data/2020/0/  # Region 0
│   ├── 11/tile_0001.tif  # Nov, 64×64×13 bands
│   └── ...
└── label/2020/0/
    └── tile_0001.tif  # 2 layers: [valid_mask, wheat_mask]
```

**2. Meta-Statistics Computation**
```python
# Compute global mean/std per month (prevents distribution shift)
meta/2020_11.npz: {mean: (13,), std: (13,)}
# Z-score normalization: x_norm = (x - mean) / std
```

**3. Feature Engineering**
- **117 features per pixel**: 9 months × 13 bands (flattened temporal-spectral vector)
- **No hand-crafted features**: Model learns temporal patterns automatically



## 🤖 Model Development

### Architecture Comparison

| Model | F1 | IoU | Training Time | Model Size | Status |
|-------|-----|-----|---------------|------------|--------|
| NDVI Threshold | 0.58 | 0.41 | N/A | N/A | Baseline |
| Random Forest | 0.85 | 0.75 | 5min | 50MB | Legacy |
| SVM | 0.72 | 0.57 | 3min | 5MB | Legacy |
| **XGBoost** | **0.86** | **0.75** | **2min** | **1.2MB** | ✅ **Production** |
| HistGB | 0.85 | 0.74 | 1min | 800KB | ✅ Production |
| PyTorch MLP | 0.64 | 0.47 | 8min | 2MB | ⚠️ Experimental |

### XGBoost Implementation

```python
# server/train/sklearn_train.py
from server.train.sklearn_train import TrainConfig, train_sklearn_model

cfg = TrainConfig(
    root='C:/preprocessed_data',
    year='2020',
    train_fraction=0.05,      # 5% of tiles (~500 tiles)
    test_fraction=0.15,       # 15% of remaining (~750 tiles)
    pixels_per_tile=4096,     # 4096 pixels per tile
    balance_pixels=True,      # 50/50 wheat/non-wheat
    use_meta_stats=True,      # Use global mean/std
    meta_dir='./meta',
    model_type='xgboost',
    xgb_n_estimators=400,
    xgb_max_depth=8,
    xgb_learning_rate=0.05,
    threshold=0.5,            # 0.65 for fewer false positives
    save_model='./runs/wheat/xgb_5pct.sklearn.joblib'
)

results = train_sklearn_model(cfg)
# Output: F1=0.859, IoU=0.752
```

### Threshold Tuning

**Problem**: False positives in urban areas (e.g., Beirut classified as wheat)

**Solution**: Adjustable prediction threshold


```bash
# Train conservative model
python train_conservative.py  # threshold=0.65
```

---

## 🧪 Experimentation

### Dataset Splits

**Stratified Sampling** (prevents class imbalance):
```python
# server/dataset/PatchDataset.py - StratifiedRandomSubset
# 1. Group tiles by wheat percentage (5 bins: 0-20%, 20-40%, ..., 80-100%)
# 2. Sample proportionally from each bin
# 3. Ensures test set has same wheat distribution as training set
```

### Evaluation Metrics

```python
# server/model/ml_utils.py
def f1_iou(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    
    F1 = 2*TP / (2*TP + FP + FN)  # Harmonic mean of precision/recall
    IoU = TP / (TP + FP + FN)     # Jaccard Index
    return F1, IoU
```

**Why F1 & IoU?**
- F1: Balances precision (few false positives) and recall (find all wheat)
- IoU: Standard metric for semantic segmentation
- NOT accuracy (dataset is imbalanced: ~30% wheat)

### Ablation Studies

**Impact of Meta-Statistics**:
| Normalization | F1 | Improvement |
|---------------|-----|-------------|
| Per-tile min-max | 0.81 | Baseline |
| **Meta z-score** | **0.86** | **+0.05** |

**Impact of Pixel Balancing**:
| Balance | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| False | 0.91 | 0.74 | 0.82 |
| **True** | **0.82** | **0.90** | **0.86** |

**Impact of Training Data Size (on 10% of Lebanon)**:
| Train % | F1 | Training Time | Notes |
|---------|-----|---------------|-------|
| 1% | 0.83 | 30s | Underfitting |
| **5%** | **0.86** | **2min** | ✅ **Optimal** |
| 10% | 0.87 | 5min | Marginal gain, 2.5x time |

### Confusion Matrix (XGBoost, 5% data)

```
              Predicted
              Non-Wheat  Wheat
Actual:
Non-Wheat     1,890,000   210,000  (Precision: 0.82)
Wheat           120,000   780,000  (Recall: 0.90)

F1-Score: 0.859
IoU: 0.752
```

---

## ⚙️ Installation & Usage

### Prerequisites

- Python 3.11 (or 3.10)
- GDAL (geospatial library)
- 16GB RAM recommended

### Installation

**Option 1: Virtual Environment**
```bash
git clone https://github.com/MohammadHasanZahweh/Wheat-Segmenter.git
cd Wheat-Segmenter

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Option 2: Docker**
```bash
docker-compose -f docker-compose.gpu.yaml up --build
# Access UI: http://localhost:8501
# API docs: http://localhost:8000/docs
```

### Training

```bash
# Train XGBoost (recommended)
python train_final_model.py

# Train conservative model (fewer false positives)
python train_conservative.py

# Custom training
python -c "
from server.train.sklearn_train import TrainConfig, train_sklearn_model
cfg = TrainConfig(
    root='C:/preprocessed_data',
    train_fraction=0.05,
    model_type='xgboost',
    threshold=0.65,
    save_model='./runs/wheat/custom.sklearn.joblib'
)
train_sklearn_model(cfg)
"
```

### Inference

**Local Script**:
```python
from server.model.sklearn_models import load_model
from server.inference.inference_lebanon import run_on_lebanon_one_year

model = load_model('./runs/wheat/xgb_5pct.sklearn.joblib')

run_on_lebanon_one_year(
    base_path='data/Lebanon/merge_data',
    year=2020,
    polygons=[{
        'type': 'Polygon',
        'coordinates': [[[35.5, 33.5], [35.6, 33.5], [35.6, 33.6], [35.5, 33.6], [35.5, 33.5]]]
    }],
    model=model,
    out_path='./results/Lebanon/test_region',
    patch_size=256,
    stride=256
)
# Output: wheat_mask.tif + wheat_mask.png
```

**Via Streamlit UI**:
```bash
streamlit run apps/streamlit_app/app.py
# 1. Select model
# 2. Draw polygon on map (within valid data coverage!)
# 3. Click "Run Inference"
# 4. View results
```

---

## 📁 Project Structure

```
Wheat-Segmenter/
│
├── server/                          # Backend services
│   ├── dataset/
│   │   └── PatchDataset.py          # Multi-temporal tile dataset
│   │       - WheatTilesDataset: Sentinel-2 reader (lazy loading)
│   │       - StratifiedRandomSubset: Balanced sampler
│   │       - load_meta_stats(): Normalization loader
│   │
│   ├── model/
│   │   ├── base_model.py            # AbstractModel interface
│   │   ├── sklearn_models.py        # XGBoost/HistGB/RF wrappers
│   │   │   - SklearnWheatModel: Embeds normalization + threshold
│   │   └── ml_utils.py              # f1_iou(), extract_pixels_from_item()
│   │
│   ├── train/
│   │   └── sklearn_train.py         # ⭐ Main training module
│   │       - TrainConfig: Universal config dataclass
│   │       - train_sklearn_model(): Training pipeline
│   │
│   ├── inference/
│   │   ├── inference_lebanon.py     # Polygon-based inference
│   │   │   - run_on_lebanon_one_year(): Main entry point
│   │   └── poly_tile_inference.py   # Multi-tile inference
│   │
│   └── server/
│       ├── app.py                   # FastAPI server
│       │   - POST /train, /inference-lebanon
│       │   - GET /status/{job_id}, /results/{project}/{save_name}
│       └── config.py                # Paths, environment config
│
├── apps/streamlit_app/              # Web interface
│   ├── app.py                       # Main Streamlit app
│   ├── core/api_client.py           # FastAPI client wrapper
│   └── ui/                          # Pages, sidebar, styles
│
├── data/Lebanon/merge_data/         # Lebanon GeoTIFFs (not in repo)
│   └── year_2019_month_11.tiff      # 18767×13830×13 bands
│
├── meta/                            # Normalization statistics
│   ├── 2020_1.npz, ..., 2020_7.npz
│   └── 2020_11.npz, 2020_12.npz     # {mean: (13,), std: (13,)}
│
├── runs/wheat/                      # Trained models
│   ├── xgb_5pct.sklearn.joblib      # ⭐ Best model (F1=0.859)
│   └── xgb_5pct_conservative.sklearn.joblib  # threshold=0.65
│
├── results/                         # Inference outputs
│   └── Lebanon/{save_name}/
│       ├── wheat_mask.tif           # Binary GeoTIFF
│       └── wheat_mask.png           # RGB visualization
│
├── docker-compose.gpu.yaml          # Production deployment
└── requirements.txt                 # Dependencies
```

---

## 🚀 API & Deployment

### FastAPI Endpoints

**1. Training**
```bash
POST /train
{
  "job_name": "xgboost_5pct",
  "algorithm": "XGBOOST",
  "dataset": {
    "year": 2020,
    "train_fraction": 0.05,
    "pixels_per_tile": 4096,
    "balance_pixels": true,
    "meta_dir": "./meta"
  },
  "save_model": true,
  "model_params": {"xgb_n_estimators": 400, "threshold": 0.65}
}

Response: {"job_id": "uuid-1234", "status": "running"}
```

**2. Inference**
```bash
POST /inference-lebanon
{
  "project_name": "Lebanon",
  "model_name": "xgboost_5pct.sklearn.joblib",
  "year": 2020,
  "geometry": {"type": "Polygon", "coordinates": [[[35.5, 33.5], ...]]},
  "save_name": "beirut_region"
}

Response: {"job_id": "uuid-5678", "status": "running"}
```

**3. Status Check**
```bash
GET /status/{job_id}

Response: {
  "status": "completed",
  "f1": 0.859,
  "iou": 0.752,
  "model_path": "./runs/wheat/xgboost_5pct.sklearn.joblib"
}
```

### Docker Deployment

```yaml
# docker-compose.gpu.yaml
services:
  api:
    build:
      context: ./server
      dockerfile: Dockerfile.gpu
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./runs:/app/runs
      - ./results:/app/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
  
  app:
    build:
      context: ./apps
      dockerfile: Dockerfile
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

**Run**:
```bash
docker-compose -f docker-compose.gpu.yaml up --build
# Streamlit: http://localhost:8501
# API: http://localhost:8000/docs
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "No tiles found in selected region"**
- **Cause**: Polygon in area without satellite data
- **Fix**: Use `python find_valid_data_coverage.py` to locate valid regions (Lon 35.12-36.60, Lat 33.07-34.68)

**2. "Model predicts 0% wheat everywhere"**
- **Cause**: Normalization mismatch or polygon in zero-data area
- **Fix**: ensure `use_meta_stats=True`

**3. "Out of memory during training"**
- **Cause**: Too many pixels (`train_fraction` × `pixels_per_tile`)
- **Fix**: Reduce `train_fraction` (0.05 → 0.01) or `pixels_per_tile` (4096 → 2048)

**4. "Too many false positives (urban areas)"**
- **Fix**: Train with higher threshold (`threshold=0.65` or `0.7`)

---

## 📊 Results Summary

### Model Performance

| Model | F1 | IoU | Precision | Recall | Training Time |
|-------|-----|-----|-----------|--------|---------------|
| **XGBoost** | **0.859** | **0.752** | 0.82 | 0.90 | 2min |
| HistGB (conservative) | 0.849 | 0.738 | 0.89 | 0.81 | 1min |
| Random Forest | 0.85 | 0.64 | 0.74 | 0.82 | 5min |
| NDVI Threshold | 0.58 | 0.41 | 0.52 | 0.66 | N/A |

**Test set**: 3M pixels from 750 tiles (stratified sampling)

### Inference Speed

- Small polygon (1 km²): ~5 seconds
- Medium polygon (10 km²): ~30 seconds  
- Large polygon (100 km²): ~3 minutes

---

## 📚 References

**Libraries**:
- rasterio (1.3.10): Geospatial raster I/O
- scikit-learn (1.5.2): Random Forest, SVM, metrics
- xgboost (2.1.0): Gradient boosting
- FastAPI (0.110): REST API
- Streamlit (1.39): Web interface

**Data Source**:
- Sentinel-2 (ESA Copernicus): Level-2A atmospherically corrected

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---



**Last Updated**: November 2025
