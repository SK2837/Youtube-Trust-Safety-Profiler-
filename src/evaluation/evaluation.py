"""
Trust & Safety evaluation framework with T&S-specific metrics.

Implements:
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Precision / Recall
- Expected Harm Score
- Threshold optimization
- Per-class evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class TrustSafetyEvaluator:
    """
    Trust & Safety-specific evaluation metrics.
    
    Prioritizes false negative reduction (safety) over false positive reduction (UX).
    """
    
    def __init__(self, fn_weight: float = 10.0, fp_weight: float = 1.0):
        """
        Initialize evaluator.
        
        Args:
            fn_weight: Weight for false negatives (harm weight)
            fp_weight: Weight for false positives (UX cost weight)
        """
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive T&S metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        
        # Precision and Recall
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = 0.0
        
        # Expected Harm Score
        n_abuse = np.sum(y_true == 1)
        n_benign = np.sum(y_true == 0)
        
        expected_harm = (
            self.fn_weight * fnr * n_abuse +
            self.fp_weight * fpr * n_benign
        )
        
        metrics = {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'fpr': fpr,
            'fnr': fnr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'expected_harm': expected_harm
        }
        
        return metrics
    
    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: np.ndarray = None
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold minimizing Expected Harm.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            thresholds: Array of thresholds to try (default: 0.1 to 0.9)
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_optimal)
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.91, 0.05)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        # Find threshold with minimum expected harm
        min_harm_idx = np.argmin([r['expected_harm'] for r in results])
        optimal_threshold = results[min_harm_idx]['threshold']
        optimal_metrics = results[min_harm_idx]
        
        return optimal_threshold, optimal_metrics, results
    
    def plot_threshold_analysis(
        self,
        threshold_results: List[Dict],
        save_path: str = None
    ):
        """
        Plot threshold sensitivity analysis.
        
        Args:
            threshold_results: List of metrics at different thresholds
            save_path: Path to save figure
        """
        df = pd.DataFrame(threshold_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # FPR and FNR
        ax = axes[0, 0]
        ax.plot(df['threshold'], df['fpr'], label='FPR', marker='o')
        ax.plot(df['threshold'], df['fnr'], label='FNR', marker='s')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.set_title('False Positive Rate vs False Negative Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision and Recall
        ax = axes[0, 1]
        ax.plot(df['threshold'], df['precision'], label='Precision', marker='o')
        ax.plot(df['threshold'], df['recall'], label='Recall', marker='s')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision vs Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Expected Harm
        ax = axes[1, 0]
        ax.plot(df['threshold'], df['expected_harm'], marker='o', color='red')
        optimal_idx = df['expected_harm'].idxmin()
        ax.axvline(df.loc[optimal_idx, 'threshold'], color='green', linestyle='--', 
                   label=f"Optimal: {df.loc[optimal_idx, 'threshold']:.2f}")
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Expected Harm')
        ax.set_title('Expected Harm Score (FN weighted 10x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 Score
        ax = axes[1, 1]
        ax.plot(df['threshold'], df['f1'], marker='o', color='purple')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score by Threshold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Threshold analysis saved to {save_path}")
        else:
            plt.show()
    
    def plot_roc_pr_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = None
    ):
        """
        Plot ROC and Precision-Recall curves.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        ax = axes[0]
        ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Recall)')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        ax = axes[1]
        ax.plot(recall, precision, linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC/PR curves saved to {save_path}")
        else:
            plt.show()
    
    def create_evaluation_table(
        self,
        metrics: Dict,
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Create formatted evaluation table.
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save table (markdown)
            
        Returns:
            DataFrame with formatted metrics
        """
        table_data = [
            ['True Positives', metrics['true_positives']],
            ['True Negatives', metrics['true_negatives']],
            ['False Positives', metrics['false_positives']],
            ['False Negatives', metrics['false_negatives']],
            ['', ''],
            ['False Positive Rate (FPR)', f"{metrics['fpr']:.4f}"],
            ['False Negative Rate (FNR)', f"{metrics['fnr']:.4f}"],
            ['True Positive Rate (TPR/Recall)', f"{metrics['tpr']:.4f}"],
            ['', ''],
            ['Precision', f"{metrics['precision']:.4f}"],
            ['Recall', f"{metrics['recall']:.4f}"],
            ['F1 Score', f"{metrics['f1']:.4f}"],
            ['AUC', f"{metrics['auc']:.4f}"],
            ['', ''],
            ['Expected Harm Score', f"{metrics['expected_harm']:.2f}"],
            ['FN Weight', f"{self.fn_weight}x"],
            ['FP Weight', f"{self.fp_weight}x"],
        ]
        
        df = pd.DataFrame(table_data, columns=['Metric', 'Value'])
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("# Statistical Evaluation Results\n\n")
                f.write(df.to_markdown(index=False))
            print(f"✓ Evaluation table saved to {save_path}")
        
        return df


def run_evaluation():
    """Run comprehensive evaluation on trained models."""
    print("\n" + "="*60)
    print("RUNNING TRUST & SAFETY EVALUATION")
    print("="*60)
    
    # Load synthetic behavioral data for evaluation
    behavioral_df = pd.read_csv("data/synthetic/unified_behavioral_features.csv")
    
    feature_cols = [
        'message_velocity',
        'burst_score',
        'account_age_days',
        'activity_ratio',
        'reply_density'
    ]
    
    X = behavioral_df[feature_cols]
    y_true = behavioral_df['is_abusive'].values
    
    # Load trained model
    import pickle
    with open('models/behavioral_risk_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Initialize evaluator
    evaluator = TrustSafetyEvaluator(fn_weight=10.0, fp_weight=1.0)
    
    # Optimize threshold
    print("\nOptimizing threshold...")
    optimal_threshold, optimal_metrics, all_results = evaluator.optimize_threshold(
        y_true, y_pred_proba
    )
    
    print(f"✓ Optimal threshold: {optimal_threshold:.3f}")
    print(f"  Expected Harm: {optimal_metrics['expected_harm']:.2f}")
    print(f"  FPR: {optimal_metrics['fpr']:.4f}")
    print(f"  FNR: {optimal_metrics['fnr']:.4f}")
    print(f"  Precision: {optimal_metrics['precision']:.4f}")
    print(f"  Recall: {optimal_metrics['recall']:.4f}")
    
    # Create output directory
    output_dir = Path("evaluation_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_threshold_analysis(all_results, 
                                     str(output_dir / "threshold_analysis.png"))
    evaluator.plot_roc_pr_curves(y_true, y_pred_proba,
                                str(output_dir / "roc_pr_curves.png"))
    
    # Create evaluation table
    eval_table = evaluator.create_evaluation_table(optimal_metrics,
                                                   str(output_dir / "evaluation_table.md"))
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(eval_table.to_string(index=False))
    
    print(f"\n✓ All outputs saved to {output_dir}/")
    
    return optimal_metrics


def main():
    """Main evaluation execution."""
    run_evaluation()


if __name__ == "__main__":
    main()
