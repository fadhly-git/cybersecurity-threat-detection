"""
Generate realistic cybersecurity attack dataset with learnable patterns.
This creates synthetic data where each attack type has distinct characteristics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_realistic_attack_data(n_samples=40000):
    """Generate cybersecurity data with REAL patterns for each attack type."""
    
    data = []
    attack_types = ['DDoS', 'Intrusion', 'Malware']
    n_per_class = n_samples // len(attack_types)
    
    start_time = datetime(2023, 1, 1)
    
    for attack_type in attack_types:
        for i in range(n_per_class):
            if attack_type == 'DDoS':
                # DDoS: High packet count, low anomaly score, specific ports
                source_port = np.random.choice([80, 443, 8080, 53], p=[0.4, 0.3, 0.2, 0.1])
                dest_port = np.random.choice([80, 443, 8080], p=[0.5, 0.3, 0.2])
                packet_length = np.random.randint(50, 200)  # Small packets
                anomaly_score = np.random.uniform(70, 95)  # High anomaly
                protocol = np.random.choice(['TCP', 'UDP'], p=[0.7, 0.3])
                traffic_type = np.random.choice(['HTTP', 'DNS'], p=[0.7, 0.3])
                severity = np.random.choice(['High', 'Critical'], p=[0.6, 0.4])
                
            elif attack_type == 'Intrusion':
                # Intrusion: Various ports, medium packets, medium-high anomaly
                source_port = np.random.randint(20000, 65000)
                dest_port = np.random.choice([22, 23, 3389, 21], p=[0.4, 0.2, 0.3, 0.1])  # SSH, Telnet, RDP, FTP
                packet_length = np.random.randint(200, 800)  # Medium packets
                anomaly_score = np.random.uniform(60, 85)  # Medium-high
                protocol = np.random.choice(['TCP', 'SSH'], p=[0.7, 0.3])
                traffic_type = np.random.choice(['SSH', 'FTP', 'Telnet'], p=[0.5, 0.3, 0.2])
                severity = np.random.choice(['High', 'Medium'], p=[0.5, 0.5])
                
            else:  # Malware
                # Malware: Random patterns, large packets, variable anomaly
                source_port = np.random.randint(30000, 60000)
                dest_port = np.random.randint(1024, 49152)
                packet_length = np.random.randint(800, 1500)  # Large packets
                anomaly_score = np.random.uniform(40, 75)  # Medium anomaly
                protocol = np.random.choice(['TCP', 'HTTP', 'HTTPS'], p=[0.5, 0.3, 0.2])
                traffic_type = np.random.choice(['HTTP', 'HTTPS', 'DNS'], p=[0.4, 0.4, 0.2])
                severity = np.random.choice(['Medium', 'Low', 'High'], p=[0.5, 0.3, 0.2])
            
            # Add some noise
            packet_length += np.random.randint(-20, 20)
            anomaly_score += np.random.uniform(-5, 5)
            anomaly_score = np.clip(anomaly_score, 0, 100)
            
            # Generate timestamp
            timestamp = start_time + timedelta(seconds=i*60)
            
            data.append({
                'Timestamp': timestamp,
                'Source_Port': source_port,
                'Destination_Port': dest_port,
                'Protocol': protocol,
                'Packet_Length': packet_length,
                'Traffic_Type': traffic_type,
                'Anomaly_Score': round(anomaly_score, 2),
                'Severity_Level': severity,
                'Attack_Type': attack_type
            })
    
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


if __name__ == '__main__':
    print("Generating realistic cybersecurity attack dataset...")
    print("=" * 80)
    
    df = generate_realistic_attack_data(40000)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nAttack type distribution:")
    print(df['Attack_Type'].value_counts())
    
    print(f"\nFeature statistics by Attack Type:")
    print(df.groupby('Attack_Type')[['Source_Port', 'Packet_Length', 'Anomaly_Score']].mean())
    
    # Save dataset
    output_path = 'data/raw/realistic_attacks.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset saved to: {output_path}")
    
    # Show correlations
    print("\n" + "=" * 80)
    print("Checking pattern strength...")
    for attack in ['DDoS', 'Intrusion', 'Malware']:
        df_temp = df.copy()
        df_temp['target'] = (df_temp['Attack_Type'] == attack).astype(int)
        corr = df_temp[['Source_Port', 'Destination_Port', 'Packet_Length', 'Anomaly_Score', 'target']].corr()['target'].abs()
        print(f"\n{attack} - Top correlations:")
        print(corr.sort_values(ascending=False).head())
