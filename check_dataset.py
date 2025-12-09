import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/cybersecurity_attacks.csv')

print("=" * 80)
print("DATASET ANALYSIS - Checking for patterns")
print("=" * 80)

# Check correlation for each attack type
for attack in ['DDoS', 'Intrusion', 'Malware']:
    df_temp = df.copy()
    df_temp['target'] = (df_temp['Attack Type'] == attack).astype(int)
    
    numeric_cols = ['Source Port', 'Destination Port', 'Packet Length', 'Anomaly Scores']
    corr = df_temp[numeric_cols + ['target']].corr()['target'].abs().sort_values(ascending=False)
    
    print(f"\n{attack} correlations:")
    print(corr.head())

# Check categorical features
print("\n" + "=" * 80)
print("Categorical feature distributions by Attack Type:")
print("=" * 80)

for col in ['Protocol', 'Packet Type', 'Traffic Type', 'Attack Signature', 'Severity Level']:
    if col in df.columns:
        print(f"\n{col}:")
        ct = pd.crosstab(df['Attack Type'], df[col], normalize='index')
        print(ct.head())
