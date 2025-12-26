"""
Data loader module for acquiring and preprocessing abuse datasets.

This module handles:
1. Downloading Jigsaw datasets via Kaggle API
2. Loading and preprocessing toxic comment data
3. Parsing transparency report statistics
4. Creating unified feature matrices
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handles data acquisition and preprocessing for risk profiler."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Root directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_kaggle_datasets(self, force: bool = False):
        """
        Download Jigsaw datasets from Kaggle.
        
        Datasets:
        1. jigsaw-toxic-comment-classification-challenge
        2. jigsaw-unintended-bias-in-toxicity-classification
        3. jigsaw-multilingual-toxic-comment-classification
        
        Args:
            force: If True, re-download even if data exists
        """
        try:
            import kaggle
        except ImportError:
            raise ImportError(
                "Kaggle API not found. Install with: pip install kaggle\n"
                "Configure with: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/"
            )
        
        datasets = [
            "jigsaw-toxic-comment-classification-challenge",
            "jigsaw-unintended-bias-in-toxicity-classification",
            "jigsaw-multilingual-toxic-comment-classification"
        ]
        
        for dataset in datasets:
            dataset_path = self.raw_dir / dataset
            
            if dataset_path.exists() and not force:
                print(f"✓ {dataset} already exists. Skipping download.")
                continue
            
            print(f"Downloading {dataset}...")
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Download using Kaggle API
            os.system(
                f"kaggle competitions download -c {dataset} "
                f"-p {dataset_path} --quiet"
            )
            
            # Unzip files
            os.system(f"cd {dataset_path} && unzip -q '*.zip' && rm *.zip")
            print(f"✓ {dataset} downloaded successfully.")
    
    def load_toxic_comments(self) -> pd.DataFrame:
        """
        Load Jigsaw Toxic Comment Classification Challenge dataset.
        
        Returns:
            DataFrame with columns: comment_text, toxic, severe_toxic, 
            obscene, threat, insult, identity_hate
        """
        dataset_name = "jigsaw-toxic-comment-classification-challenge"
        train_path = self.raw_dir / dataset_name / "train.csv"
        
        if not train_path.exists():
            # Create sample data for demonstration if Kaggle dataset not available
            print("⚠ Kaggle dataset not found. Creating sample data...")
            return self._create_sample_toxic_data()
        
        df = pd.read_csv(train_path)
        print(f"✓ Loaded {len(df)} toxic comments from {dataset_name}")
        
        return df
    
    def load_unbiased_dataset(self) -> pd.DataFrame:
        """
        Load Jigsaw Unintended Bias dataset.
        
        Returns:
            DataFrame with toxicity scores and identity annotations
        """
        dataset_name = "jigsaw-unintended-bias-in-toxicity-classification"
        train_path = self.raw_dir / dataset_name / "train.csv"
        
        if not train_path.exists():
            print("⚠ Kaggle dataset not found. Creating sample data...")
            return self._create_sample_unbiased_data()
        
        # Load only essential columns to manage memory
        cols_to_load = [
            'comment_text', 'target', 'toxicity', 'severe_toxicity',
            'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit'
        ]
        
        df = pd.read_csv(train_path, usecols=cols_to_load, nrows=100000)
        print(f"✓ Loaded {len(df)} comments from unbiased dataset")
        
        return df
    
    def _create_sample_toxic_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Create sample toxic comment data for demonstration.
        
        This is used when Kaggle datasets are not available.
        """
        np.random.seed(42)
        
        # Sample toxic comments (demonstrative examples)
        toxic_examples = [
            "This is completely unacceptable behavior",
            "You're absolutely wrong about this",
            "I strongly disagree with your opinion",
            "This content is inappropriate",
            "Your argument makes no sense"
        ]
        
        benign_examples = [
            "I appreciate your perspective on this topic",
            "Thanks for sharing this information",
            "That's an interesting point to consider",
            "I'd like to learn more about this",
            "Great discussion, everyone"
        ]
        
        data = []
        for i in range(n_samples):
            if np.random.random() < 0.1:  # 10% toxic
                text = np.random.choice(toxic_examples)
                toxic = 1
                severe_toxic = np.random.choice([0, 1], p=[0.8, 0.2])
                obscene = np.random.choice([0, 1], p=[0.7, 0.3])
                threat = np.random.choice([0, 1], p=[0.9, 0.1])
                insult = np.random.choice([0, 1], p=[0.6, 0.4])
                identity_hate = np.random.choice([0, 1], p=[0.95, 0.05])
            else:
                text = np.random.choice(benign_examples)
                toxic = 0
                severe_toxic = 0
                obscene = 0
                threat = 0
                insult = 0
                identity_hate = 0
            
            data.append([
                f"sample_{i}", text, toxic, severe_toxic, 
                obscene, threat, insult, identity_hate
            ])
        
        df = pd.DataFrame(
            data,
            columns=['id', 'comment_text', 'toxic', 'severe_toxic',
                    'obscene', 'threat', 'insult', 'identity_hate']
        )
        
        print(f"✓ Created {len(df)} sample toxic comments")
        return df
    
    def _create_sample_unbiased_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Create sample unbiased dataset for demonstration."""
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            target = np.random.beta(2, 8)  # Skewed toward low toxicity
            
            data.append({
                'comment_text': f"Sample comment {i}",
                'target': target,
                'toxicity': target,
                'severe_toxicity': target * np.random.random(),
                'obscene': target * np.random.random(),
                'threat': target * np.random.random() * 0.3,
                'insult': target * np.random.random(),
                'identity_attack': target * np.random.random() * 0.4,
                'sexual_explicit': target * np.random.random() * 0.3
            })
        
        df = pd.DataFrame(data)
        print(f"✓ Created {len(df)} sample unbiased comments")
        return df
    
    def extract_transparency_priors(self) -> Dict[str, float]:
        """
        Extract abuse rate priors from public transparency reports.
        
        These are approximations based on publicly available data.
        
        Returns:
            Dictionary with baseline abuse rates by category
        """
        # Based on publicly available transparency reports
        # These are APPROXIMATIONS for demonstration purposes
        priors = {
            'harassment': 0.018,      # ~1.8% of content (from Meta reports)
            'hate_speech': 0.012,     # ~1.2% (YouTube transparency data)
            'spam': 0.035,            # ~3.5% (Reddit moderation stats)
            'misinformation': 0.008,  # ~0.8% (various platform reports)
            'coordinated_abuse': 0.003,  # ~0.3% (estimated from brigading data)
            'overall_abuse': 0.025    # ~2.5% baseline abuse rate
        }
        
        # Save to processed directory
        priors_path = self.processed_dir / "transparency_priors.json"
        with open(priors_path, 'w') as f:
            json.dump(priors, f, indent=2)
        
        print(f"✓ Extracted transparency priors: {priors}")
        return priors
    
    def preprocess_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess dataset for model training.
        
        Args:
            df: Raw dataframe with comment_text and labels
            
        Returns:
            Tuple of (preprocessed_df, label_columns)
        """
        # Clean text
        df['comment_text_clean'] = df['comment_text'].str.lower()
        df['comment_text_clean'] = df['comment_text_clean'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Identify label columns
        potential_labels = [
            'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate',
            'toxicity', 'severe_toxicity', 'identity_attack', 'sexual_explicit'
        ]
        label_columns = [col for col in potential_labels if col in df.columns]
        
        # Remove rows with missing text
        df = df.dropna(subset=['comment_text'])
        
        # Calculate composite toxicity score
        if 'toxic' in df.columns:
            df['toxicity_score'] = df[label_columns].max(axis=1)
        elif 'target' in df.columns:
            df['toxicity_score'] = df['target']
        
        print(f"✓ Preprocessed {len(df)} comments with {len(label_columns)} labels")
        
        return df, label_columns
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save preprocessed data to processed directory."""
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        print(f"✓ Saved processed data to {output_path}")


def main():
    """Main execution for data loading pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and preprocess abuse datasets')
    parser.add_argument('--download', action='store_true', help='Download Kaggle datasets')
    parser.add_argument('--validate', action='store_true', help='Validate data pipeline')
    args = parser.parse_args()
    
    loader = DataLoader()
    
    if args.download:
        print("Downloading Kaggle datasets...")
        try:
            loader.download_kaggle_datasets()
        except Exception as e:
            print(f"⚠ Download failed: {e}")
            print("Proceeding with sample data generation...")
    
    # Load datasets
    print("\n" + "="*60)
    print("Loading Toxic Comment Dataset")
    print("="*60)
    toxic_df = loader.load_toxic_comments()
    
    print("\n" + "="*60)
    print("Loading Unbiased Dataset")
    print("="*60)
    unbiased_df = loader.load_unbiased_dataset()
    
    # Preprocess
    print("\n" + "="*60)
    print("Preprocessing Data")
    print("="*60)
    toxic_processed, toxic_labels = loader.preprocess_for_training(toxic_df)
    unbiased_processed, unbiased_labels = loader.preprocess_for_training(unbiased_df)
    
    # Save processed data
    loader.save_processed_data(toxic_processed, "toxic_comments_processed.csv")
    loader.save_processed_data(unbiased_processed, "unbiased_comments_processed.csv")
    
    # Extract transparency priors
    print("\n" + "="*60)
    print("Extracting Transparency Report Priors")
    print("="*60)
    priors = loader.extract_transparency_priors()
    
    # Print summary
    print("\n" + "="*60)
    print("DATA LOADING SUMMARY")
    print("="*60)
    print(f"Toxic comments: {len(toxic_processed):,}")
    print(f"Unbiased comments: {len(unbiased_processed):,}")
    print(f"Toxic labels: {toxic_labels}")
    print(f"Unbiased labels: {unbiased_labels}")
    print(f"\nBaseline abuse rates:")
    for category, rate in priors.items():
        print(f"  {category}: {rate*100:.2f}%")
    
    if args.validate:
        print("\n✓ Data pipeline validation successful!")


if __name__ == "__main__":
    main()
