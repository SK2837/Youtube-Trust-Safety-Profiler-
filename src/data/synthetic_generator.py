"""
Synthetic behavioral signal generator for live Q&A feature.

This module generates realistic behavioral patterns for:
- Message velocity (messages/second)
- Burst detection features
- User account characteristics
- Reply graph density (coordination signals)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import json


class SyntheticBehavioralGenerator:
    """Generates synthetic behavioral signals for risk modeling."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
    
    def generate_message_velocity(
        self,
        n_users: int = 1000,
        stream_duration_mins: int = 60
    ) -> pd.DataFrame:
        """
        Generate message velocity patterns.
        
        Benign users: ~0.5-2 messages/min (exponential distribution)
        Abusive users: ~5-20 messages/min (higher rate, potential spam/flood)
        
        Args:
            n_users: Number of users to simulate
            stream_duration_mins: Duration of live stream
            
        Returns:
            DataFrame with user_id, message_velocity, is_abusive
        """
        data = []
        
        for user_id in range(n_users):
            # 10% abusive users
            is_abusive = np.random.random() < 0.1
            
            if is_abusive:
                # Higher message velocity for abusers
                velocity = np.random.exponential(scale=10) + 5  # 5-20 msgs/min
            else:
                # Normal users post less frequently
                velocity = np.random.exponential(scale=1.5)  # 0.5-2 msgs/min
            
            data.append({
                'user_id': f'user_{user_id}',
                'message_velocity': velocity,
                'is_abusive': int(is_abusive)
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_burst_patterns(
        self,
        n_events: int = 500
    ) -> pd.DataFrame:
        """
        Generate burst detection features.
        
        Burst Score: Measures message arrival concentration
        - Low score (0-0.3): Organic, distributed arrival
        - Medium (0.3-0.7): Elevated activity
        - High (0.7-1.0): Coordinated flooding/spam
        
        Args:
            n_events: Number of message events to simulate
            
        Returns:
            DataFrame with event_id, burst_score, is_coordinated
        """
        data = []
        
        for event_id in range(n_events):
            # 15% coordinated abuse events
            is_coordinated = np.random.random() < 0.15
            
            if is_coordinated:
                # High burst score for coordinated attacks
                burst_score = np.random.beta(8, 2)  # Skewed toward 1.0
            else:
                # Low burst score for organic activity
                burst_score = np.random.beta(2, 8)  # Skewed toward 0.0
            
            data.append({
                'event_id': f'event_{event_id}',
                'burst_score': burst_score,
                'is_coordinated': int(is_coordinated)
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_account_features(
        self,
        n_accounts: int = 1000
    ) -> pd.DataFrame:
        """
        Generate user account features.
        
        Key signals:
        - Account age (days)
        - Activity history (total posts)
        - Account age vs. activity ratio (new accounts with high activity = suspicious)
        
        Args:
            n_accounts: Number of accounts to simulate
            
        Returns:
            DataFrame with account features and abuse labels
        """
        data = []
        
        for account_id in range(n_accounts):
            # 12% throwaway/abusive accounts
            is_throwaway = np.random.random() < 0.12
            
            if is_throwaway:
                # New accounts with disproportionate activity
                account_age_days = np.random.exponential(scale=15) + 1  # 1-30 days
                total_activity = np.random.exponential(scale=200) + 50  # High activity
            else:
                # Established accounts with normal activity
                account_age_days = np.random.exponential(scale=365) + 30  # 30+ days
                total_activity = np.random.exponential(scale=100)
            
            # Activity-to-age ratio (suspicious if high for new accounts)
            activity_ratio = total_activity / max(account_age_days, 1)
            
            data.append({
                'account_id': f'account_{account_id}',
                'account_age_days': account_age_days,
                'total_activity': total_activity,
                'activity_ratio': activity_ratio,
                'is_throwaway': int(is_throwaway)
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_reply_graph_density(
        self,
        n_message_groups: int = 300
    ) -> pd.DataFrame:
        """
        Generate reply graph density features.
        
        Coordination signal: High reply graph density indicates
        coordinated behavior (users replying to each other rapidly)
        
        Args:
            n_message_groups: Number of message groups to simulate
            
        Returns:
            DataFrame with graph density and coordination labels
        """
        data = []
        
        for group_id in range(n_message_groups):
            # 20% coordinated reply networks
            is_coordinated_network = np.random.random() < 0.20
            
            if is_coordinated_network:
                # High density: coordinated brigading
                reply_density = np.random.beta(7, 2)  # 0.6-0.9
                unique_users = np.random.randint(5, 15)
                reply_count = np.random.randint(20, 50)
            else:
                # Low density: organic discussion
                reply_density = np.random.beta(2, 5)  # 0.1-0.4
                unique_users = np.random.randint(10, 50)
                reply_count = np.random.randint(5, 30)
            
            data.append({
                'group_id': f'group_{group_id}',
                'reply_density': reply_density,
                'unique_users': unique_users,
                'reply_count': reply_count,
                'is_coordinated_network': int(is_coordinated_network)
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_unified_behavioral_features(
        self,
        n_samples: int = 5000
    ) -> pd.DataFrame:
        """
        Generate unified behavioral feature set combining all signals.
        
        This creates a realistic dataset for training behavioral risk model.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with all behavioral features and abuse label
        """
        data = []
        
        for sample_id in range(n_samples):
            # Determine if this is an abusive pattern (25% abuse rate in behavioral data)
            is_abusive = np.random.random() < 0.25
            
            if is_abusive:
                # Abusive behavioral profile
                message_velocity = np.random.exponential(scale=8) + 4
                burst_score = np.random.beta(7, 2)
                account_age_days = np.random.exponential(scale=20) + 1
                activity_ratio = np.random.exponential(scale=15) + 5
                reply_density = np.random.beta(6, 3)
            else:
                # Benign behavioral profile
                message_velocity = np.random.exponential(scale=1.5)
                burst_score = np.random.beta(2, 7)
                account_age_days = np.random.exponential(scale=300) + 30
                activity_ratio = np.random.exponential(scale=2)
                reply_density = np.random.beta(2, 6)
            
            data.append({
                'sample_id': f'sample_{sample_id}',
                'message_velocity': message_velocity,
                'burst_score': burst_score,
                'account_age_days': account_age_days,
                'activity_ratio': activity_ratio,
                'reply_density': reply_density,
                'is_abusive': int(is_abusive)
            })
        
        df = pd.DataFrame(data)
        return df
    
    def save_synthetic_data(self, output_dir: str = "data/synthetic"):
        """
        Generate and save all synthetic behavioral datasets.
        
        Args:
            output_dir: Directory to save synthetic data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating synthetic behavioral data...")
        
        # Generate individual feature datasets
        velocity_df = self.generate_message_velocity()
        burst_df = self.generate_burst_patterns()
        account_df = self.generate_account_features()
        reply_df = self.generate_reply_graph_density()
        
        # Save individual datasets
        velocity_df.to_csv(output_path / "message_velocity.csv", index=False)
        burst_df.to_csv(output_path / "burst_patterns.csv", index=False)
        account_df.to_csv(output_path / "account_features.csv", index=False)
        reply_df.to_csv(output_path / "reply_graph_density.csv", index=False)
        
        print(f"✓ Saved message velocity: {len(velocity_df)} samples")
        print(f"✓ Saved burst patterns: {len(burst_df)} samples")
        print(f"✓ Saved account features: {len(account_df)} samples")
        print(f"✓ Saved reply graph density: {len(reply_df)} samples")
        
        # Generate unified feature set for model training
        unified_df = self.generate_unified_behavioral_features(n_samples=10000)
        unified_df.to_csv(output_path / "unified_behavioral_features.csv", index=False)
        print(f"✓ Saved unified behavioral features: {len(unified_df)} samples")
        
        # Save metadata
        metadata = {
            'generator_seed': self.seed,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'datasets': {
                'message_velocity': len(velocity_df),
                'burst_patterns': len(burst_df),
                'account_features': len(account_df),
                'reply_graph_density': len(reply_df),
                'unified_behavioral': len(unified_df)
            },
            'assumptions': {
                'baseline_abuse_rate': 0.10,
                'coordinated_attack_rate': 0.15,
                'throwaway_account_rate': 0.12,
                'description': 'All behavioral signals are synthetically generated based on threat modeling assumptions'
            }
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ All synthetic data saved to {output_path}")
        print(f"\n⚠ IMPORTANT: These are simulated behavioral patterns, not real YouTube data")


def main():
    """Main execution for synthetic data generation."""
    generator = SyntheticBehavioralGenerator(seed=42)
    generator.save_synthetic_data()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SYNTHETIC DATA GENERATION SUMMARY")
    print("="*60)
    
    # Load and display sample statistics
    unified_df = pd.read_csv("data/synthetic/unified_behavioral_features.csv")
    
    print(f"\nUnified Behavioral Features:")
    print(f"  Total samples: {len(unified_df):,}")
    print(f"  Abusive patterns: {unified_df['is_abusive'].sum():,} ({unified_df['is_abusive'].mean()*100:.1f}%)")
    print(f"\nFeature Statistics:")
    print(unified_df.describe()[['message_velocity', 'burst_score', 'account_age_days', 'activity_ratio', 'reply_density']])


if __name__ == "__main__":
    main()
