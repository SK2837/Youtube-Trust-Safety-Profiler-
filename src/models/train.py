"""
Model training pipeline for risk profiler.

Trains:
1. Behavioral risk classifier on synthetic data
2. Prepares content classifier (using pre-trained BERT)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data.data_loader import DataLoader
from data.synthetic_generator import SyntheticBehavioralGenerator
from models.risk_model import BehavioralRiskClassifier, UnifiedRiskScorer
import pickle


def train_behavioral_model():
    """Train behavioral risk classifier."""
    print("="*60)
    print("TRAINING BEHAVIORAL RISK MODEL")
    print("="*60)
    
    # Generate synthetic behavioral data if not exists
    synthetic_path = Path("data/synthetic/unified_behavioral_features.csv")
    
    if not synthetic_path.exists():
        print("\nGenerating synthetic behavioral data...")
        generator = SyntheticBehavioralGenerator(seed=42)
        generator.save_synthetic_data()
    
    # Load synthetic data
    print("\nLoading synthetic behavioral data...")
    behavioral_df = pd.read_csv(synthetic_path)
    
    print(f"✓ Loaded {len(behavioral_df):,} behavioral samples")
    print(f"  Abusive: {behavioral_df['is_abusive'].sum():,} ({behavioral_df['is_abusive'].mean()*100:.1f}%)")
    
    # Prepare features and labels
    feature_cols = [
        'message_velocity',
        'burst_score',
        'account_age_days',
        'activity_ratio',
        'reply_density'
    ]
    
    X = behavioral_df[feature_cols]
    y = behavioral_df['is_abusive'].values
    
    # Train model
    classifier = BehavioralRiskClassifier()
    metrics = classifier.train(X, y, test_size=0.2)
    
    # Print results
    print("\n" + "="*60)
    print("BEHAVIORAL MODEL TRAINING RESULTS")
    print("="*60)
    print(f"AUC Score: {metrics['auc']:.4f}")
    print("\nFeature Importance:")
    for feature, importance in sorted(
        metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {feature:20s}: {importance:.4f}")
    
    # Save model
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    classifier.save_model("models/behavioral_risk_classifier.pkl")
    
    return classifier, metrics


def prepare_content_data():
    """Load and prepare content data for evaluation."""
    print("\n" + "="*60)
    print("PREPARING CONTENT DATA")
    print("="*60)
    
    loader = DataLoader()
    
    # Load toxic comments
    toxic_df = loader.load_toxic_comments()
    toxic_processed, label_cols = loader.preprocess_for_training(toxic_df)
    
    # Save processed data
    loader.save_processed_data(toxic_processed, "toxic_comments_processed.csv")
    
    return toxic_processed


def main():
    """Main training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train risk profiler models')
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" "*20 + "RISK PROFILER TRAINING PIPELINE")
    print("="*70)
    
    # Train behavioral model
    behavioral_classifier, behavioral_metrics = train_behavioral_model()
    
    # Prepare content data
    content_df = prepare_content_data()
    
    # Initialize unified scorer
    print("\n" + "="*60)
    print("INITIALIZING UNIFIED RISK SCORER")
    print("="*60)
    
    scorer = UnifiedRiskScorer()
    scorer.behavioral_classifier = behavioral_classifier
    
    print("✓ Unified risk scorer initialized")
    print(f"  Content weight: {scorer.content_weight}")
    print(f"  Behavioral weight: {scorer.behavioral_weight}")
    
    # Save metadata
    metadata = {
        'behavioral_auc': behavioral_metrics['auc'],
        'feature_importance': behavioral_metrics['feature_importance'],
        'content_weight': scorer.content_weight,
        'behavioral_weight': scorer.behavioral_weight,
        'thresholds': {
            'low': 0.3,
            'medium': 0.7,
            'high': 1.2
        }
    }
    
    import json
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Model metadata saved to models/model_metadata.json")
    
    if args.eval:
        print("\n" + "="*60)
        print("Running evaluation...")
        print("="*60)
        from evaluation.evaluation import run_evaluation
        run_evaluation()
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
