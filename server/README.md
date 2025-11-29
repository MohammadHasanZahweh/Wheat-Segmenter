# Server - Backend Services (Docker)

The `server/` directory contains all backend components for wheat segmentation: datasets, models, inference engines, and the FastAPI service.

**Note**: This documentation focuses on Docker deployment. For end-users, simply use `docker-compose up` from the project root - no manual setup required.

---

## 📁 Directory Structure

```
server/
├── dataset/              # Data loading and preprocessing
│   ├── PatchDataset.py      # Multi-temporal tile dataset
│   ├── PixelDataset.py      # Pixel-level dataset
│   └── TileDataset.py       # Tile utilities
│
├── model/               # Model implementations
│   ├── base_model.py        # Abstract model interface
│   ├── sklearn_models.py    # XGBoost/HistGB/RF wrappers
│   ├── torch_pixel_model.py # PyTorch MLP
│   ├── RNNmodel.py          # RNN temporal model
│   └── ml_utils.py          # F1/IoU metrics utilities
│
├── inference/           # Inference engines
│   ├── inference_lebanon.py # Polygon-based inference
│   ├── tile_inference.py    # Single tile inference
│   ├── poly_tile_inference.py # Multi-tile inference
│   └── data_to_rgb.py       # RGB visualization
│
├── server/              # FastAPI backend
│   ├── app.py               # Main API application
│   ├── config.py            # Configuration
│   └── utils.py             # Utilities
│
├── Dockerfile.gpu       # GPU Docker image  
└── requirements.txt     # Backend dependencies
```

---
## 🐳 Docker Deployment

### GPU Image (Recommended)

```bash
# From project root
docker-compose -f docker-compose.gpu.yaml up --build
```

**Exposes:**
- FastAPI: http://localhost:8000
- API Docs: http://localhost:8000/docs

### CPU Image

```bash
# From project root
docker-compose up --build
```

---

## 🚀 FastAPI Endpoints

### Health Check
```bash
GET /
```
Returns API status and available endpoints.

### Run Inference
```bash
POST /inference-lebanon
# Server - Backend Services (Docker)
{
  "project_name": "lebanon_2020",
  "model_name": "xgboost_5pct_threshold_0.5.sklearn.joblib",
  "geometry": {
    "type": "Polygon",
```
**Response:**
```json
{
  "job_id": "abc123",
  "status": "running"
}
```

### Check Status
```bash
GET /status/{job_id}
```

**Response:**
```json
{
  "job_id": "abc123",
  "status": "completed",
  "output_path": "/results/Lebanon/my_region/wheat_mask.tif"
}
```

### List Models
```bash
GET /models/list
```

**Response:**
```json
{
  "models": [
    "xgboost_5pct_threshold_0.5.sklearn.joblib",
    "xgboost_5pct_threshold_0.6.sklearn.joblib",
    "xgboost_5pct_threshold_0.7.sklearn.joblib"
  ]


## 📦 Components
#### **sklearn_models.py**
Pre-trained XGBoost models for wheat segmentation.

**Available Models** (in `runs/wheat/`):
- `xgboost_5pct_threshold_0.5.sklearn.joblib` - Balanced (F1=0.86)
- `xgboost_5pct_threshold_0.6.sklearn.joblib` - Fewer false positives
- `xgboost_5pct_threshold_0.65.sklearn.joblib` - Conservative
- `xgboost_5pct_threshold_0.7.sklearn.joblib` - Very conservative

All models include:
- ✅ Z-score normalization (using `meta/*.npz` statistics)
- ✅ Configurable probability threshold
- ✅ 117 features (9 months × 13 bands)

### 2. Inference (`inference/`)

#### **inference_lebanon.py**
Main inference engine for polygon-based wheat detection.

**Function**: `run_on_lebanon_one_year()`

**Features:**
- Loads Sentinel-2 tiles from `data/Lebanon/merge_data/`
- Supports GeoJSON, Shapely, and GeoDataFrame inputs
- Tile-based processing (256×256 patches)
- Outputs GeoTIFF wheat mask

### 3. Dataset (`dataset/`)

#### **PatchDataset.py**
Multi-temporal Sentinel-2 tile loader.

**Features:**
- Lazy loading of 64×64 tiles
- Z-score normalization using global statistics
- Returns: `{"x": [T, B, H, W], "valid": [H, W], "wheat": [H, W]}`

---

## 🔧 Configuration

### Environment Variables

Set in `docker-compose.gpu.yaml`:

```yaml
environment:
  - DATA_DIR=/data          # Satellite imagery
  - RUNS_DIR=/runs          # Model files
  - RESULTS_DIR=/results    # Inference outputs
```

### Model Selection

Models are automatically loaded from `RUNS_DIR/wheat/`.

To use a different model, specify `model_name` in API request:
```json
{
  "model_name": "xgboost_5pct_threshold_0.65.sklearn.joblib"
}
```

---

## 📊 Architecture

### Data Flow

```
User request (polygon) → FastAPI → 
Load satellite tiles → Normalize → 
Model prediction → Generate mask → 
Save GeoTIFF → Return result
```

### Normalization Pipeline

1. Load global statistics from `meta/2020_*.npz` (per-month mean/std)
2. Apply Z-score: `x_norm = (x - mean) / std`
3. Ensure consistent normalization between training and inference

---

## 🐛 Troubleshooting

### "Model not found"
**Fix**: Ensure models are in `runs/wheat/` folder (should be included in repository).

### "No data for region"
**Fix**: Polygon must be within Lebanon coverage area (Lon 35.5-36.5, Lat 33.5-34.5).

### "Out of memory"
**Fix**: Reduce polygon size or use smaller `patch_size` in inference config.

---

## 📚 Additional Resources

- **Main README:** `../README.md` - Quick start guide
- **Streamlit App:** `../apps/README.md` - Web interface documentation
- **API Documentation:** http://localhost:8000/docs (when server running)
