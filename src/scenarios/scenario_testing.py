"""
Adversarial scenario testing and stress simulation.

Implements 3 misuse scenarios:
1. Coordinated harassment during political live stream
2. Spam flooding at peak concurrency
3. Targeted abuse amplification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import json


class ScenarioSimulator:
    """Simulates adversarial abuse scenarios for stress testing."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize scenario simulator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
    
    def scenario_1_coordinated_harassment(
        self,
        duration_mins: int = 10,
        abuse_volume: int = 500,
        num_attackers: int = 50
    ) -> pd.DataFrame:
        """
        Scenario 1: Coordinated Harassment During Political Live Stream.
        
        Pattern:
        - Burst arrivals from coordinated accounts
        - Identity-based attacks targeting guest speaker
        - High message velocity and reply density
        
        Args:
            duration_mins: Scenario duration
            abuse_volume: Number of abusive messages
            num_attackers: Number of coordinated attackers
            
        Returns:
            DataFrame with simulated messages
        """
        print("\n" + "="*60)
        print("SCENARIO 1: Coordinated Harassment")
        print("="*60)
        print(f"Duration: {duration_mins} minutes")
        print(f"Abusive messages: {abuse_volume}")
        print(f"Coordinated attackers: {num_attackers}")
        
        messages = []
        
        # Simulate burst arrivals in first 2 minutes
        for i in range(abuse_volume):
            # Concentrated in first 20% of time
            if np.random.random() < 0.6:
                timestamp_mins = np.random.uniform(0, duration_mins * 0.2)
            else:
                timestamp_mins = np.random.uniform(0, duration_mins)
            
            # Assign to coordinated attackers
            attacker_id = f"attacker_{np.random.randint(0, num_attackers)}"
            
            # High toxicity content
            content_risk = np.random.beta(8, 2)  # Skewed toward 1.0
            
            # Behavioral signals for coordinated attack
            message_velocity = np.random.exponential(scale=15) + 10
            burst_score = np.random.beta(9, 1)  # Very high burst
            account_age_days = np.random.exponential(scale=10) + 1  # New accounts
            activity_ratio = np.random.exponential(scale=20) + 10  # High ratio
            reply_density = np.random.beta(8, 2)  # High coordination
            
            messages.append({
                'scenario': 'coordinated_harassment',
                'timestamp_mins': timestamp_mins,
                'user_id': attacker_id,
                'is_abusive': 1,
                'content_risk': content_risk,
                'message_velocity': message_velocity,
                'burst_score': burst_score,
                'account_age_days': account_age_days,
                'activity_ratio': activity_ratio,
                'reply_density': reply_density,
                'is_live': True,
                'creator_subscribers': 2_000_000,
                'is_sensitive_topic': True
            })
        
        # Add some benign messages
        for i in range(int(abuse_volume * 0.3)):
            timestamp_mins = np.random.uniform(0, duration_mins)
            
            messages.append({
                'scenario': 'coordinated_harassment',
                'timestamp_mins': timestamp_mins,
                'user_id': f"user_{i}",
                'is_abusive': 0,
                'content_risk': np.random.beta(2, 8),
                'message_velocity': np.random.exponential(scale=1.5),
                'burst_score': np.random.beta(2, 8),
                'account_age_days': np.random.exponential(scale=300) + 30,
                'activity_ratio': np.random.exponential(scale=2),
                'reply_density': np.random.beta(2, 6),
                'is_live': True,
                'creator_subscribers': 2_000_000,
                'is_sensitive_topic': True
            })
        
        df = pd.DataFrame(messages).sort_values('timestamp_mins').reset_index(drop=True)
        
        print(f"✓ Generated {len(df)} messages ({df['is_abusive'].sum()} abusive)")
        
        return df
    
    def scenario_2_spam_flooding(
        self,
        duration_mins: int = 5,
        spam_volume: int = 2000,
        spam_rate_per_min: int = 400
    ) -> pd.DataFrame:
        """
        Scenario 2: Spam Flooding at Peak Concurrency.
        
        Pattern:
        - High volume link injection and promotional content
        - Sustained high message velocity
        - Low account age, high activity
        
        Args:
            duration_mins: Scenario duration
            spam_volume: Number of spam messages
            spam_rate_per_min: Spam arrival rate
            
        Returns:
            DataFrame with simulated messages
        """
        print("\n" + "="*60)
        print("SCENARIO 2: Spam Flooding")
        print("="*60)
        print(f"Duration: {duration_mins} minutes")
        print(f"Spam messages: {spam_volume}")
        print(f"Rate: {spam_rate_per_min} msgs/min")
        
        messages = []
        
        # Spam messages
        for i in range(spam_volume):
            timestamp_mins = np.random.uniform(0, duration_mins)
            
            # Spam characteristics
            content_risk = np.random.beta(4, 6)  # Medium toxicity (promotional)
            message_velocity = np.random.exponential(scale=20) + 15
            burst_score = np.random.beta(7, 3)
            account_age_days = np.random.exponential(scale=5) + 1
            activity_ratio = np.random.exponential(scale=25) + 15
            reply_density = np.random.beta(3, 7)  # Lower (not coordinated, just spam)
            
            messages.append({
                'scenario': 'spam_flooding',
                'timestamp_mins': timestamp_mins,
                'user_id': f"spammer_{np.random.randint(0, 200)}",
                'is_abusive': 1,
                'content_risk': content_risk,
                'message_velocity': message_velocity,
                'burst_score': burst_score,
                'account_age_days': account_age_days,
                'activity_ratio': activity_ratio,
                'reply_density': reply_density,
                'is_live': True,
                'creator_subscribers': 2_000_000,
                'is_sensitive_topic': False
            })
        
        # Add legitimate users
        for i in range(int(spam_volume * 0.2)):
            timestamp_mins = np.random.uniform(0, duration_mins)
            
            messages.append({
                'scenario': 'spam_flooding',
                'timestamp_mins': timestamp_mins,
                'user_id': f"user_{i}",
                'is_abusive': 0,
                'content_risk': np.random.beta(2, 8),
                'message_velocity': np.random.exponential(scale=1.5),
                'burst_score': np.random.beta(2, 8),
                'account_age_days': np.random.exponential(scale=300) + 30,
                'activity_ratio': np.random.exponential(scale=2),
                'reply_density': np.random.beta(2, 6),
                'is_live': True,
                'creator_subscribers': 2_000_000,
                'is_sensitive_topic': False
            })
        
        df = pd.DataFrame(messages).sort_values('timestamp_mins').reset_index(drop=True)
        
        print(f"✓ Generated {len(df)} messages ({df['is_abusive'].sum()} spam)")
        
        return df
    
    def scenario_3_targeted_abuse(
        self,
        duration_mins: int = 15,
        abuse_volume: int = 200,
        escalation_rate: float = 0.7
    ) -> pd.DataFrame:
        """
        Scenario 3: Targeted Abuse with Escalation.
        
        Pattern:
        - Iterative toxicity escalation over time
        - Coordinated replies forming abuse network
        - Targeting creator with marginalized identity
        
        Args:
            duration_mins: Scenario duration
            abuse_volume: Number of abusive messages
            escalation_rate: Rate of toxicity escalation
            
        Returns:
            DataFrame with simulated messages
        """
        print("\n" + "="*60)
        print("SCENARIO 3: Targeted Abuse with Escalation")
        print("="*60)
        print(f"Duration: {duration_mins} minutes")
        print(f"Abusive messages: {abuse_volume}")
        print(f"Escalation rate: {escalation_rate}")
        
        messages = []
        
        # Simulate escalating abuse
        for i in range(abuse_volume):
            # Time-based escalation
            timestamp_mins = np.random.uniform(0, duration_mins)
            escalation_factor = min(1.0, timestamp_mins / duration_mins + 0.3)
            
            # Escalating toxicity
            base_toxicity = np.random.beta(5, 5)
            content_risk = min(1.0, base_toxicity * escalation_factor * 1.5)
            
            # Coordinated reply network
            message_velocity = np.random.exponential(scale=10) + 5
            burst_score = np.random.beta(6, 4)
            account_age_days = np.random.exponential(scale=15) + 1
            activity_ratio = np.random.exponential(scale=12) + 6
            reply_density = np.random.beta(7, 3)  # High reply coordination
            
            messages.append({
                'scenario': 'targeted_abuse',
                'timestamp_mins': timestamp_mins,
                'user_id': f"attacker_{np.random.randint(0, 30)}",
                'is_abusive': 1,
                'content_risk': content_risk,
                'message_velocity': message_velocity,
                'burst_score': burst_score,
                'account_age_days': account_age_days,
                'activity_ratio': activity_ratio,
                'reply_density': reply_density,
                'escalation_factor': escalation_factor,
                'is_live': True,
                'creator_subscribers': 500_000,
                'is_sensitive_topic': True
            })
        
        # Add benign messages
        for i in range(int(abuse_volume * 0.4)):
            timestamp_mins = np.random.uniform(0, duration_mins)
            
            messages.append({
                'scenario': 'targeted_abuse',
                'timestamp_mins': timestamp_mins,
                'user_id': f"user_{i}",
                'is_abusive': 0,
                'content_risk': np.random.beta(2, 8),
                'message_velocity': np.random.exponential(scale=1.5),
                'burst_score': np.random.beta(2, 8),
                'account_age_days': np.random.exponential(scale=300) + 30,
                'activity_ratio': np.random.exponential(scale=2),
                'reply_density': np.random.beta(2, 6),
                'escalation_factor': 0.0,
                'is_live': True,
                'creator_subscribers': 500_000,
                'is_sensitive_topic': True
            })
        
        df = pd.DataFrame(messages).sort_values('timestamp_mins').reset_index(drop=True)
        
        print(f"✓ Generated {len(df)} messages ({df['is_abusive'].sum()} abusive)")
        
        return df
    
    def analyze_scenario_metrics(
        self,
        scenario_df: pd.DataFrame,
        risk_scores: np.ndarray,
        threshold: float = 0.7
    ) -> Dict:
        """
        Analyze detection metrics for scenario.
        
        Args:
            scenario_df: Scenario data
            risk_scores: Predicted risk scores
            threshold: Detection threshold
            
        Returns:
            Dictionary with scenario metrics
        """
        scenario_name = scenario_df['scenario'].iloc[0]
        
        y_true = scenario_df['is_abusive'].values
        y_pred = (risk_scores >= threshold).astype(int)
        
        # Detection latency (time to first detection)
        abusive_detected = scenario_df[
            (scenario_df['is_abusive'] == 1) & (risk_scores >= threshold)
        ]
        
        if len(abusive_detected) > 0:
            first_detection_time = abusive_detected['timestamp_mins'].min()
            detection_latency_p50 = abusive_detected['timestamp_mins'].median()
            detection_latency_p90 = abusive_detected['timestamp_mins'].quantile(0.9)
        else:
            first_detection_time = None
            detection_latency_p50 = None
            detection_latency_p90 = None
        
        # False negative concentration (FN in first 2 minutes)
        early_period = scenario_df['timestamp_mins'] <= 2.0
        early_abuse = scenario_df[early_period & (scenario_df['is_abusive'] == 1)]
        early_misses = early_abuse[risk_scores[early_period & (scenario_df['is_abusive'] == 1)] < threshold]
        
        fn_early_concentration = len(early_misses) / len(early_abuse) if len(early_abuse) > 0 else 0
        
        # Moderator load (messages flagged per minute)
        flagged = (risk_scores >= threshold).sum()
        duration = scenario_df['timestamp_mins'].max()
        moderator_load_per_min = flagged / duration if duration > 0 else 0
        
        # Confusion matrix
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            'scenario': scenario_name,
            'total_messages': len(scenario_df),
            'abusive_messages': y_true.sum(),
            'detected': tp,
            'missed': fn,
            'false_positives': fp,
            'fnr': fnr,
            'fpr': fpr,
            'first_detection_mins': first_detection_time,
            'detection_latency_p50_mins': detection_latency_p50,
            'detection_latency_p90_mins': detection_latency_p90,
            'fn_early_concentration': fn_early_concentration,
            'moderator_load_per_min': moderator_load_per_min,
            'total_moderator_review': flagged
        }
        
        return metrics


def run_all_scenarios():
    """Run all three adversarial scenarios."""
    print("\n" + "="*70)
    print(" " * 20 + "ADVERSARIAL SCENARIO TESTING")
    print("="*70)
    
    simulator = ScenarioSimulator(seed=42)
    
    # Generate scenarios
    scenario_1 = simulator.scenario_1_coordinated_harassment()
    scenario_2 = simulator.scenario_2_spam_flooding()
    scenario_3 = simulator.scenario_3_targeted_abuse()
    
    # Save scenario data
    output_dir = Path("scenario_outputs")
    output_dir.mkdir(exist_ok=True)
    
    scenario_1.to_csv(output_dir / "scenario_1_coordinated_harassment.csv", index=False)
    scenario_2.to_csv(output_dir / "scenario_2_spam_flooding.csv", index=False)
    scenario_3.to_csv(output_dir / "scenario_3_targeted_abuse.csv", index=False)
    
    print("\n✓ All scenarios saved to scenario_outputs/")
    
    # Simulate risk scoring (using simplified scoring for demonstration)
    from models.risk_model import UnifiedRiskScorer, ContextualRiskMultiplier
    
    scorer = UnifiedRiskScorer()
    
    results = []
    
    for scenario_name, scenario_df in [
        ('Scenario 1', scenario_1),
        ('Scenario 2', scenario_2),
        ('Scenario 3', scenario_3)
    ]:
        print(f"\n{'-'*60}")
        print(f"Analyzing {scenario_name}")
        print(f"{'-'*60}")
        
        # Calculate risk scores (simplified: weighted combination)
        behavioral_risk = (
            scenario_df['message_velocity'] / 30 +
            scenario_df['burst_score'] +
            (1 / (scenario_df['account_age_days'] + 1)) * 10 +
            scenario_df['activity_ratio'] / 40 +
            scenario_df['reply_density']
        ) / 5
        behavioral_risk = np.clip(behavioral_risk, 0, 1)
        
        content_risk = scenario_df['content_risk'].values
        
        # Contextual multiplier
        context = {
            'is_live': scenario_df['is_live'].iloc[0],
            'creator_subscribers': scenario_df['creator_subscribers'].iloc[0],
            'is_sensitive_topic': scenario_df['is_sensitive_topic'].iloc[0]
        }
        contextual_mult = ContextualRiskMultiplier.calculate_multiplier(**context)
        
        # Unified risk score
        unified_risk = (
            scorer.content_weight * content_risk +
            scorer.behavioral_weight * behavioral_risk
        ) * contextual_mult
        
        # Analyze metrics
        metrics = simulator.analyze_scenario_metrics(scenario_df, unified_risk, threshold=0.7)
        results.append(metrics)
        
        print(f"  Total messages: {metrics['total_messages']}")
        print(f"  Abusive: {metrics['abusive_messages']}")
        print(f"  Detected: {metrics['detected']}")
        print(f"  Missed: {metrics['missed']} (FNR: {metrics['fnr']:.2%})")
        print(f"  False Positives: {metrics['false_positives']} (FPR: {metrics['fpr']:.2%})")
        if metrics['first_detection_mins']:
            print(f"  First detection: {metrics['first_detection_mins']:.2f} mins")
            print(f"  Detection latency (p50): {metrics['detection_latency_p50_mins']:.2f} mins")
        print(f"  FN early concentration: {metrics['fn_early_concentration']:.2%}")
        print(f"  Moderator load: {metrics['moderator_load_per_min']:.1f} msgs/min")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "scenario_metrics.csv", index=False)
    
    # Save as JSON for documentation
    with open(output_dir / "scenario_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ SCENARIO TESTING COMPLETE")
    print("="*70)
    print(f"Results saved to {output_dir}/")
    
    return results


def main():
    """Main execution for scenario testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run adversarial scenario testing')
    parser.add_argument('--all-scenarios', action='store_true', help='Run all 3 scenarios')
    args = parser.parse_args()
    
    if args.all_scenarios:
        run_all_scenarios()
    else:
        print("Use --all-scenarios to run all adversarial scenarios")


if __name__ == "__main__":
    main()
