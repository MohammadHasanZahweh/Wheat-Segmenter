"""
Test script for the unified training API endpoint.
Start the server first: uvicorn server.server.app:app --reload
"""
import requests
import time
import json

API_BASE = "http://localhost:8000"

def test_training_job(algorithm="xgboost", use_meta_stats=True):
    """Submit a training job and poll for results."""
    
    # Create training request
    payload = {
        "job_name": f"test_{algorithm}_{'meta' if use_meta_stats else 'no_meta'}",
        "algorithm": algorithm,
        "dataset": {
            "root": r"C:\Users\Administrator\Desktop\preprocessed_data",
            "year": "2020",
            "months": [11, 12, 1, 2, 3, 4, 5, 6, 7],
            "train_fraction": 0.01,
            "test_fraction": 0.25,
            "pixels_per_tile": 4096,
            "balance_pixels": True,
            "seed": 42,
            "use_meta_stats": use_meta_stats,
            "meta_dir": "./meta"
        },
        "save_model": True,
        "model_params": {
            "n_estimators": 100,  # Fast for testing
            "max_depth": 8
        }
    }
    
    print(f"\n{'='*60}")
    print(f"Testing {algorithm.upper()} {'WITH' if use_meta_stats else 'WITHOUT'} meta stats")
    print(f"{'='*60}")
    
    # Submit job
    response = requests.post(f"{API_BASE}/train", json=payload)
    response.raise_for_status()
    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"✓ Job submitted: {job_id}")
    
    # Poll for completion
    max_wait = 600  # 10 minutes
    start = time.time()
    while time.time() - start < max_wait:
        response = requests.get(f"{API_BASE}/train/status", params={"id": job_id})
        status_data = response.json()
        status = status_data.get("status", "unknown")
        
        if status == "completed":
            print(f"✓ Training completed!")
            print(f"\nResults:")
            print(f"  F1 Score:  {status_data.get('f1', 'N/A'):.4f}")
            print(f"  IoU:       {status_data.get('iou', 'N/A'):.4f}")
            print(f"  Precision: {status_data.get('precision', 'N/A'):.4f}")
            print(f"  Recall:    {status_data.get('recall', 'N/A'):.4f}")
            print(f"  Model:     {status_data.get('model_path', 'N/A')}")
            return status_data
        elif status == "failed":
            print(f"✗ Training failed: {status_data.get('error', 'Unknown error')}")
            return None
        elif status == "running":
            print(f"⏳ Training in progress... ({int(time.time() - start)}s elapsed)")
            time.sleep(10)
        else:
            print(f"? Unknown status: {status}")
            time.sleep(5)
    
    print(f"✗ Timeout after {max_wait}s")
    return None


if __name__ == "__main__":
    # Test health endpoint
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Server health: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Server not running. Start with:")
        print("  uvicorn server.server.app:app --reload")
        exit(1)
    
    # Test all algorithms
    algorithms = ["xgboost", "hist_gradient_boosting", "random_forest"]
    
    for algo in algorithms:
        # Test WITH meta stats
        result_meta = test_training_job(algo, use_meta_stats=True)
        
        # Test WITHOUT meta stats
        result_no_meta = test_training_job(algo, use_meta_stats=False)
        
        # Compare
        if result_meta and result_no_meta:
            f1_improvement = result_meta['f1'] - result_no_meta['f1']
            iou_improvement = result_meta['iou'] - result_no_meta['iou']
            print(f"\n📊 Meta Stats Impact for {algo}:")
            print(f"   F1 improvement:  {f1_improvement:+.4f} ({f1_improvement/result_no_meta['f1']*100:+.1f}%)")
            print(f"   IoU improvement: {iou_improvement:+.4f} ({iou_improvement/result_no_meta['iou']*100:+.1f}%)")
