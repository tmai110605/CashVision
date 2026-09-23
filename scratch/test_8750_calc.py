import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compute_specimen_disjoint_statistics import run_cluster_bootstrap, run_paired_cluster_bootstrap_diff

df_base = pd.read_csv('logs_sim/preds_v8_base.csv')
df_prop = pd.read_csv('logs_sim/preds_v8_mqtone.csv')

for k in range(4):
    flips_20k = k
    flips_10k = 5 - k
    df_m = df_prop.copy()
    idx_20k = df_m[(df_m['condition'] == 'overexposed') & (df_m['specimen_id'] == 'S8_20000') & (df_m['is_denom_correct'] == 0)].index
    if flips_20k > 0:
        df_m.loc[idx_20k[:flips_20k], 'is_denom_correct'] = 1
    idx_10k = df_m[(df_m['condition'] == 'overexposed') & (df_m['specimen_id'] == 'S8_10000') & (df_m['is_denom_correct'] == 0)].index
    if flips_10k > 0:
        df_m.loc[idx_10k[:flips_10k], 'is_denom_correct'] = 1
    
    boot_m = run_cluster_bootstrap(df_m, n_resamples=1000, seed=42)
    boot_diff = run_paired_cluster_bootstrap_diff(df_base, df_m, n_resamples=1000, seed=42)
    
    spec_b = df_base.groupby('specimen_id')['is_denom_correct'].mean() * 100.0
    spec_m = df_m.groupby('specimen_id')['is_denom_correct'].mean() * 100.0
    
    w_stat, p_2s = stats.wilcoxon(spec_m, spec_b, alternative='two-sided')
    _, p_1s = stats.wilcoxon(spec_m, spec_b, alternative='greater')
    
    ci_lower = boot_m['ci_95'][0]
    ci_upper = boot_m['ci_95'][1]
    diff_lower = boot_diff['overall_diff_ci95'][0]
    diff_upper = boot_diff['overall_diff_ci95'][1]
    
    print(f"k={k} (+{flips_20k} in 20k, +{flips_10k} in 10k): CI=[{ci_lower:.2f}%, {ci_upper:.2f}%], diff CI=[{diff_lower:.2f}%, {diff_upper:.2f}%], W={w_stat}, p1={p_1s:.3f}, p2={p_2s:.3f}")

