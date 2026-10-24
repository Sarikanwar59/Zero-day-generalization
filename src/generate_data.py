import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 3000

normal = pd.DataFrame({
    "duration": rng.exponential(1.5, n),
    "src_bytes": rng.lognormal(7, 1, n),
    "dst_bytes": rng.lognormal(8, 1, n),
    "packet_count": rng.poisson(20, n),
    "failed_connections": rng.poisson(0.2, n),
    "label": 0
})

attack = pd.DataFrame({
    "duration": rng.exponential(0.3, n),
    "src_bytes": rng.lognormal(5, 1.5, n),
    "dst_bytes": rng.lognormal(4, 1.5, n),
    "packet_count": rng.poisson(65, n),
    "failed_connections": rng.poisson(5, n),
    "label": 1
})

df = pd.concat([normal, attack], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/network_traffic.csv", index=False)

print("Dataset created successfully")
print("Shape:", df.shape)
print(df["label"].value_counts())
