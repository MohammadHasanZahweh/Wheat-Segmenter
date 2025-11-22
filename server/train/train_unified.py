#!/usr/bin/env python
"""
CLI wrapper for consolidated sklearn training.

Backward-compatible with original train_xgboost.py, train_histgb.py, etc.
Now uses the unified sklearn_train module.
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from server.train.sklearn_train import TrainConfig, train_sklearn_model


def parse_args():
    parser = argparse.ArgumentParser(description="Unified sklearn model trainer for wheat segmentation")
    
    # Model selection
    parser.add_argument("--model", type=str, required=True, 
                       choices=["xgboost", "histgb", "random_forest", "svm"],
                       help="Model type to train")
    
    # Data arguments
    parser.add_argument("--root", required=True, help="Preprocessed root containing data/ and label/")
    parser.add_argument("--year", required=True, help="Year subfolder under data/ and label/")
    parser.add_argument("--regions", nargs="*", default=None, help="Region ids (strings). If omitted, use all.")
    parser.add_argument("--months", nargs="*", type=int, default=[11,12,1,2,3,4,5,6,7], help="Months order")
    
    # Sampling arguments
    parser.add_argument("--train-fraction", type=float, default=0.01, help="Fraction of ALL tiles for training")
    parser.add_argument("--test-fraction", type=float, default=0.25, help="Fraction of REMAINING tiles for testing")
    parser.add_argument("--pixels-per-tile", type=int, default=4096, help="Max valid pixels sampled per tile")
    parser.add_argument("--balance-pixels", action="store_true", help="Class-balance pixel sampling within tiles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Normalization
    parser.add_argument("--use-meta-stats", action="store_true", 
                       help="Use precomputed meta statistics for normalization (./meta/)")
    parser.add_argument("--meta-dir", type=str, default="./meta", help="Path to meta directory")
    
    # XGBoost arguments
    parser.add_argument("--xgb-n-estimators", type=int, default=400, help="XGBoost n_estimators")
    parser.add_argument("--xgb-max-depth", type=int, default=8, help="XGBoost max_depth")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05, help="XGBoost learning_rate")
    parser.add_argument("--xgb-subsample", type=float, default=0.8, help="XGBoost subsample")
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8, help="XGBoost colsample_bytree")
    
    # HistGB arguments
    parser.add_argument("--hgb-max-depth", type=int, default=8, help="HistGB max_depth")
    parser.add_argument("--hgb-max-iter", type=int, default=400, help="HistGB max_iter")
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05, help="HistGB learning_rate")
    parser.add_argument("--hgb-l2-regularization", type=float, default=0.0, help="HistGB l2_regularization")
    
    # RandomForest arguments
    parser.add_argument("--rf-n-estimators", type=int, default=200, help="RandomForest n_estimators")
    parser.add_argument("--rf-max-depth", type=int, default=None, help="RandomForest max_depth")
    
    # SVM arguments
    parser.add_argument("--svm-kernel", type=str, default="rbf", help="SVM kernel")
    parser.add_argument("--svm-C", type=float, default=1.0, help="SVM C parameter")
    parser.add_argument("--svm-gamma", type=str, default="scale", help="SVM gamma")
    
    # Output
    parser.add_argument("--save-path", "--save-model", dest="save_path", default=None, 
                       help="Path to save the trained model (.joblib)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse gamma for SVM
    svm_gamma = args.svm_gamma
    try:
        svm_gamma = float(svm_gamma)
    except ValueError:
        pass  # Keep as string
    
    # Create config
    config = TrainConfig(
        # Data
        root=str(args.root),
        year=str(args.year),
        regions=[str(r) for r in args.regions] if args.regions else None,
        months=tuple(int(m) for m in args.months),
        
        # Sampling
        train_fraction=float(args.train_fraction),
        test_fraction=float(args.test_fraction),
        pixels_per_tile=int(args.pixels_per_tile),
        balance_pixels=bool(args.balance_pixels),
        seed=int(args.seed),
        
        # Normalization
        use_meta_stats=bool(args.use_meta_stats),
        meta_dir=str(args.meta_dir),
        
        # Model type
        model_type=args.model,
        
        # XGBoost
        xgb_n_estimators=int(args.xgb_n_estimators),
        xgb_max_depth=int(args.xgb_max_depth),
        xgb_learning_rate=float(args.xgb_learning_rate),
        xgb_subsample=float(args.xgb_subsample),
        xgb_colsample_bytree=float(args.xgb_colsample_bytree),
        
        # HistGB
        hgb_max_depth=int(args.hgb_max_depth) if args.hgb_max_depth else None,
        hgb_max_iter=int(args.hgb_max_iter),
        hgb_learning_rate=float(args.hgb_learning_rate),
        hgb_l2_regularization=float(args.hgb_l2_regularization),
        
        # RandomForest
        rf_n_estimators=int(args.rf_n_estimators),
        rf_max_depth=int(args.rf_max_depth) if args.rf_max_depth else None,
        
        # SVM
        svm_kernel=str(args.svm_kernel),
        svm_C=float(args.svm_C),
        svm_gamma=svm_gamma,
        
        # Output
        save_model=str(args.save_path) if args.save_path else None,
    )
    
    # Train
    results = train_sklearn_model(config)
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
