# Apps - Streamlit Web Application (Docker)

The `apps/` directory contains the Streamlit web application for wheat segmentation. It provides an interactive UI for running inference and visualizing results through a browser interface.

**Note**: This documentation focuses on Docker deployment. For end-users, simply use `docker-compose up` from the project root.

---

## 📁 Directory Structure

```
apps/
├── streamlit_app/
│   ├── app.py               # Main Streamlit application
│   ├── core/                # Core functionality
│   │   ├── api_client.py       # FastAPI client wrapper
│   │   ├── charts.py           # Visualization charts
│   │   ├── exports.py          # Export utilities (GeoTIFF, GeoJSON)
│   │   ├── geo.py              # Geographic utilities
│   │   ├── inference.py        # Local inference engine
│   │   ├── loaders.py          # Data loading utilities
│   │   └── metrics.py          # Metrics calculation
│   │
│   └── ui/                  # UI components
│       ├── pages.py            # Page renderers
│       ├── sidebar.py          # Sidebar configuration
│       └── styles.py           # CSS styles
│
├── Dockerfile           # Streamlit Docker image
└── requirements.txt     # App dependencies
```

---

## 🐳 Docker Deployment

### Full Stack Deployment

```bash
# From project root
docker-compose -f docker-compose.gpu.yaml up --build
```

**Access:**
- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8000

### Standalone Streamlit (Development)

```bash
# Build image
cd apps
docker build -t wheat-streamlit .

# Run container
docker run -p 8501:8501 \
  -e API_URL=http://localhost:8000 \
  wheat-streamlit
```

---

## 📱 Application Features

### 1. **Interactive Map**
- View Lebanon satellite data coverage
- Draw polygon regions with mouse
- See tile boundaries overlay
- Auto-zoom to selected regions

### 2. **Model Selection**
- Choose from 4 pre-trained models with different thresholds
- Models automatically loaded from `runs/wheat/` folder
- Switch models without restarting

### 3. **One-Click Inference**
- Select model → Draw region → Click "Run Inference"
- Real-time progress tracking
- Results displayed on map overlay

### 4. **Results Visualization**
- Wheat distribution heatmap
- Coverage statistics (% wheat, total area)
- Interactive result explorer

### 5. **Export Options**
- Download GeoTIFF (for GIS software)
- Download PNG (for presentations)
- Export statistics (CSV/JSON)

---

## 🎯 User Workflow

### Step-by-Step Guide

**1. Access Application**
```
Open browser → http://localhost:8501
```

**2. Select Model**
```
Sidebar → Model Selection → Choose threshold
- 0.5: Balanced (default)
- 0.65: Conservative (fewer false positives)
```

**3. Draw Region**
```
Interactive Map → Click polygon tool → Draw your area
```

**4. Run Inference**
```
Click "Run Inference" button → Wait for processing → View results
```

**5. Download Results**
```
Results section → Click "Download GeoTIFF" or "Download PNG"
```

---

## 🔌 Components

### **api_client.py**
FastAPI client for backend communication.

```python
client = TrainAPI("http://localhost:8000")

# Run inference
response = client.start_inference_lebanon(payload)

# Check status
status = client.status(job_id)
```

### **inference.py**
Local inference engine for running models in Streamlit.

**Features:**
- Pixel-level prediction
- Progress tracking
- Cancellation support

### **geo.py**
Geographic utilities for map operations.

**Functions:**
- `bounds_to_polygon()`: Convert raster bounds to Shapely polygon
- `polygon_to_geojson()`: Convert Shapely to GeoJSON

### **exports.py**
Export utilities for results.

**Functions:**
- `export_geotiff()`: Save wheat mask as GeoTIFF
- `export_geojson()`: Save polygons as GeoJSON
- `export_statistics()`: Save metrics as CSV

---

## 🗺️ Interactive Map Features

### Drawing Tools

The app uses `streamlit-folium` with `folium.plugins.Draw` for polygon drawing.

**Available Tools:**
- ✅ Polygon drawing
- ✅ Rectangle drawing
- ✅ Edit existing polygons
- ✅ Delete polygons
- ❌ No markers, circles, or polylines

**Tips:**
- Draw within Lebanon boundaries (Lon 35.5-36.5, Lat 33.5-34.5)
- Stay within visible blue tile rectangles (data coverage)
- Smaller regions process faster

---

## ⚙️ Configuration

### Environment Variables

```bash
# Set API endpoint
export API_URL=http://localhost:8000

# Set data root (optional)
export DATA_ROOT=/path/to/data
```

### Docker Compose Integration

```yaml
services:
  app:
    build:
      context: ./apps
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

---

## 🐛 Troubleshooting

### Map Not Rendering
**Cause**: JavaScript/CSS loading issue  
**Fix**: Hard refresh browser (Ctrl+Shift+R) or clear cache

### API Connection Error
**Cause**: Backend not running  
**Fix**: 
1. Check API is running: `curl http://localhost:8000/health`
2. Verify API_URL in sidebar matches backend address

### "No tiles found"
**Cause**: Polygon drawn outside data coverage  
**Fix**: Draw polygon within visible blue rectangles on map

### Inference Stuck
**Cause**: Large polygon or slow processing  
**Fix**: 
1. Click "Cancel Inference" button
2. Draw smaller region
3. Check Docker container logs: `docker logs wheat_app`

---

## 📊 UI Components

### Sidebar
- Model selection dropdown
- API URL configuration
- Inference controls (Run/Cancel)
- Job status display

### Main Panel
- Interactive Folium map
- Drawing tools
- Results visualization
- Export buttons

### Results Section
- Coverage statistics
- Wheat percentage
- Total area (hectares)
- Download options

---

## 📚 Additional Resources

- **Main README:** `../README.md` - Quick start guide
- **Server Documentation:** `../server/README.md` - Backend API
- **Streamlit Docs:** https://docs.streamlit.io
- **Folium Docs:** https://python-visualization.github.io/folium/

---

**Built for easy wheat monitoring - No coding required!**  
**Last Updated**: November 2025
