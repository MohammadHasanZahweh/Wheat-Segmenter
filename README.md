# 🌾 Wheat-Segmenter

**Wheat-Segmenter** is a computer vision project designed to automatically segment wheat fields from **Sentinel-2 satellite imagery** using both **machine learning** and **deep learning** techniques.  
This helps researchers and farmers analyze crop health, estimate yield, and monitor growth patterns efficiently. The project aims to streamline agricultural data collection and enable **precision farming** through AI-powered image analysis.

## 📝 TODO
- [ ] (@st1) Prepare a preprocessing data code
- [ ] (@Zahweh) calculate bandwise mean and std_dev (for normalization)
- [ ] (@st2) Create a random Sampler to sample 1% data of all, try to be as stratisfied as possible
- [ ] Implement baseline machine learning models (Random Forest, SVM)  
- [ ] Logistec Reg, KNN, Random Forestm, XGboost, ...
- [ ] (@Zahweh) Train self-supervised model  
- [ ] Evaluate models using IoU and F1-score  
- [ ] Visualize segmentation results and overlays (+ interface) 
- [ ] Optimize performance and add model comparison notebook  
- [ ] Write documentation and create docker
- [ ] Automate Collect and preprocess Sentinel-2 imagery  


## 🚀 Features

- Automatic segmentation of wheat fields from Sentinel-2 imagery  
- Support for both machine learning (e.g., Random Forest, SVM) and deep learning (e.g., U-Net, SegNet) models  
- Scalable preprocessing pipeline for satellite data   
- Model evaluation using metrics such as IoU, F1-score, and pixel accuracy  
- Easy visualization of segmentation results using color masks and overlay maps  


## 🧠 Tech Stack

- **Languages:** Python  
- **Libraries & Tools:**  
  - `PyTorch` – deep learning models  
  - `scikit-learn` – traditional ML algorithms  
  - `rasterio` – geospatial data handling  
  - `matplotlib`, `seaborn` – visualization  
  - `NumPy`, `Pandas` – data preprocessing and analysis  



## 📂 Project Structure

```

Wheat-Segmenter/
│
├── data/                 # Raw and processed Sentinel-2 imagery
├── notebooks/            # Jupyter notebooks for experiments
├── src/                  # Source code for models and preprocessing
│   ├── preprocessing/    # Data preparation scripts
│   ├── models/           # ML and DL architectures
│   └── utils/            # Helper functions
├── results/              # Segmentation outputs and evaluation metrics
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation

````

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Wheat-Segmenter.git
cd Wheat-Segmenter

# Create and activate a virtual environment (optional)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

---

## 🛰️ Usage

To be implmeneted later on
<!-- 1. Download Sentinel-2 imagery for your area of interest.
2. Place the images in the `data/` directory.
3. Run preprocessing:

   ```bash
   python src/preprocessing/prepare_data.py
   ```
4. Train a model:

   ```bash
   python src/models/train_unet.py
   ```
5. Visualize segmentation results:

   ```bash
   python src/utils/visualize_results.py
   ``` -->






