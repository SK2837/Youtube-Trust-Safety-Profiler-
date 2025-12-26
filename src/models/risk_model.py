"""
Multi-layer risk scoring model for YouTube Live Q&A feature.

Architecture:
1. Content Risk Score (RC): BERT-based toxicity classifier
2. Behavioral Risk Score (RB): XGBoost on behavioral signals
3. Contextual Risk Multiplier (MC): Feature-specific context
4. Unified Feature Risk Score: R = (α × RC + β × RB) × MC
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import xgboost as xgb

# Deep learning imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class ContentRiskClassifier:
    """
    Content-based toxicity classifier using BERT.
    
    Uses pre-trained models from HuggingFace fine-tuned on toxic comments.
    """
    
    def __init__(self, model_name: str = "unitary/toxic-bert"):
        """
        Initialize content risk classifier.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Content risk model loaded on {self.device}")
    
    def predict_content_risk(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Predict content risk scores for texts.
        
        Args:
            texts: List of comment texts
            batch_size: Batch size for inference
            
        Returns:
            Array of content risk scores [0, 1]
        """
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                # Get toxicity probability (assume binary classification or take max)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # If multi-label, take maximum across toxic categories
                if probs.shape[1] > 2:
                    toxic_scores = probs[:, 1:].max(dim=1)[0]
                else:
                    toxic_scores = probs[:, 1]  # Binary: non-toxic, toxic
                
                all_scores.extend(toxic_scores.cpu().numpy())
        
        return np.array(all_scores)
    
    def predict_detailed_scores(
        self,
        texts: List[str]
    ) -> pd.DataFrame:
        """
        Get detailed toxicity scores by category.
        
        Returns:
            DataFrame with per-category scores
        """
        # Simplified version - returns single toxicity score
        # Full implementation would use multi-label toxic comment classifier
        content_scores = self.predict_content_risk(texts)
        
        return pd.DataFrame({
            'content_risk_score': content_scores,
            'toxic': content_scores,
            'severe_toxic': content_scores * 0.3,  # Approximation
            'harassment': content_scores * 0.5,
            'hate_speech': content_scores * 0.4
        })


class BehavioralRiskClassifier:
    """
    Behavioral risk classifier using XGBoost.
    
    Trained on synthetic behavioral signals.
    """
    
    def __init__(self):
        """Initialize behavioral risk classifier."""
        self.model = None
        self.feature_names = [
            'message_velocity',
            'burst_score',
            'account_age_days',
            'activity_ratio',
            'reply_density'
        ]
    
    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train behavioral risk model.
        
        Args:
            X: Feature dataframe
            y: Labels (is_abusive)
            test_size: Test set proportion
            
        Returns:
            Training metrics dictionary
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        print("Training behavioral risk model...")
        self.model.fit(X_train, y_train, verbose=False)
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'auc': auc,
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        }
        
        print(f"✓ Behavioral model trained: AUC = {auc:.4f}")
        
        return metrics
    
    def predict_behavioral_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict behavioral risk scores.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of behavioral risk scores [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, path: str):
        """Save trained model."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Model loaded from {path}")


class ContextualRiskMultiplier:
    """
    Contextual risk multipliers based on feature context.
    
    Multipliers:
    - Live event status: 1.5x
    - High visibility (creator size): 1.3x
    - Sensitive topic: 1.4x
    """
    
    @staticmethod
    def calculate_multiplier(
        is_live: bool = True,
        creator_subscribers: int = 0,
        is_sensitive_topic: bool = False
    ) -> float:
        """
        Calculate contextual risk multiplier.
        
        Args:
            is_live: Whether event is live
            creator_subscribers: Creator subscriber count
            is_sensitive_topic: Political/sensitive topic flag
            
        Returns:
            Contextual multiplier (>= 1.0)
        """
        multiplier = 1.0
        
        # Live event multiplier
        if is_live:
            multiplier *= 1.5
        
        # High-visibility creator
        if creator_subscribers > 1_000_000:
            multiplier *= 1.3
        elif creator_subscribers > 100_000:
            multiplier *= 1.15
        
        # Sensitive topic
        if is_sensitive_topic:
            multiplier *= 1.4
        
        return multiplier


class UnifiedRiskScorer:
    """
    Unified risk scoring system combining all components.
    
    R_feature = (α × RC + β × RB) × MC
    
    Where:
    - RC: Content risk score
    - RB: Behavioral risk score
    - MC: Contextual multiplier
    - α = 0.6 (content weight)
    - β = 0.4 (behavioral weight)
    """
    
    def __init__(
        self,
        content_weight: float = 0.6,
        behavioral_weight: float = 0.4
    ):
        """
        Initialize unified risk scorer.
        
        Args:
            content_weight: Weight for content risk (α)
            behavioral_weight: Weight for behavioral risk (β)
        """
        self.content_weight = content_weight
        self.behavioral_weight = behavioral_weight
        
        # Initialize components
        self.content_classifier = None  # Lazy loading
        self.behavioral_classifier = BehavioralRiskClassifier()
        self.context_multiplier = ContextualRiskMultiplier()
    
    def initialize_content_classifier(self):
        """Lazy initialization of content classifier (heavy model)."""
        if self.content_classifier is None:
            self.content_classifier = ContentRiskClassifier()
    
    def calculate_unified_risk(
        self,
        content_score: float,
        behavioral_score: float,
        contextual_multiplier: float = 1.0
    ) -> float:
        """
        Calculate unified feature risk score.
        
        Args:
            content_score: Content risk score [0, 1]
            behavioral_score: Behavioral risk score [0, 1]
            contextual_multiplier: Context multiplier (>= 1.0)
            
        Returns:
            Unified risk score [0, inf), typically [0, 2]
        """
        base_risk = (
            self.content_weight * content_score +
            self.behavioral_weight * behavioral_score
        )
        
        unified_risk = base_risk * contextual_multiplier
        
        return unified_risk
    
    def classify_risk_level(self, risk_score: float) -> str:
        """
        Classify risk into categories.
        
        Args:
            risk_score: Unified risk score
            
        Returns:
            Risk level: 'low', 'medium', 'high', 'critical'
        """
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.7:
            return 'medium'
        elif risk_score < 1.2:
            return 'high'
        else:
            return 'critical'
    
    def score_batch(
        self,
        texts: List[str],
        behavioral_features: pd.DataFrame,
        context: Dict = None
    ) -> pd.DataFrame:
        """
        Score a batch of messages.
        
        Args:
            texts: List of message texts
            behavioral_features: DataFrame with behavioral signals
            context: Dictionary with contextual features
            
        Returns:
            DataFrame with all risk scores and classifications
        """
        if context is None:
            context = {'is_live': True, 'creator_subscribers': 0, 'is_sensitive_topic': False}
        
        # Initialize content classifier if needed
        self.initialize_content_classifier()
        
        # Get content risk scores
        content_scores = self.content_classifier.predict_content_risk(texts)
        
        # Get behavioral risk scores
        behavioral_scores = self.behavioral_classifier.predict_behavioral_risk(
            behavioral_features[self.behavioral_classifier.feature_names]
        )
        
        # Calculate contextual multiplier
        contextual_mult = self.context_multiplier.calculate_multiplier(**context)
        
        # Calculate unified risk scores
        unified_scores = np.array([
            self.calculate_unified_risk(c, b, contextual_mult)
            for c, b in zip(content_scores, behavioral_scores)
        ])
        
        # Classify risk levels
        risk_levels = [self.classify_risk_level(score) for score in unified_scores]
        
        results = pd.DataFrame({
            'content_risk': content_scores,
            'behavioral_risk': behavioral_scores,
            'contextual_multiplier': contextual_mult,
            'unified_risk_score': unified_scores,
            'risk_level': risk_levels
        })
        
        return results


def main():
    """Demo of risk scoring system."""
    print("="*60)
    print("RISK SCORING SYSTEM DEMO")
    print("="*60)
    
    # Create simple demo without heavy model loading
    scorer = UnifiedRiskScorer()
    
    # Demo unified risk calculation
    print("\nExample Risk Calculations:")
    print("-" * 60)
    
    scenarios = [
        {'content': 0.9, 'behavioral': 0.8, 'context': 1.5, 'desc': 'High risk: Toxic content from suspicious account during live event'},
        {'content': 0.3, 'behavioral': 0.2, 'context': 1.0, 'desc': 'Low risk: Normal comment from established user'},
        {'content': 0.6, 'behavioral': 0.7, 'context': 1.95, 'desc': 'Critical risk: Moderately toxic + high velocity + high-profile event'},
    ]
    
    for scenario in scenarios:
        risk = scorer.calculate_unified_risk(
            scenario['content'],
            scenario['behavioral'],
            scenario['context']
        )
        level = scorer.classify_risk_level(risk)
        
        print(f"\nScenario: {scenario['desc']}")
        print(f"  Content: {scenario['content']:.2f} | Behavioral: {scenario['behavioral']:.2f} | Context: {scenario['context']:.2f}x")
        print(f"  → Unified Risk: {risk:.3f} [{level.upper()}]")
    
    print("\n" + "="*60)
    print("Risk Thresholds:")
    print("  Low:      R < 0.3")
    print("  Medium:   0.3 ≤ R < 0.7")
    print("  High:     0.7 ≤ R < 1.2")
    print("  Critical: R ≥ 1.2")
    print("="*60)


if __name__ == "__main__":
    main()
