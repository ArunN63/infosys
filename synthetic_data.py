# create_synthetic_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Ensure Sample/data folder exists
os.makedirs("Sample/data", exist_ok=True)

# --- competitor_history.csv (time series) ---
start = datetime(2024, 1, 1)
rows = []
competitors = ["compA", "compB"]
products = ["prod1", "prod2", "prod3"]

for d in range(300):  # 300 days
    date = start + timedelta(days=d)
    for comp in competitors:
        for p in products:
            price = 100 + (hash(f"{comp}{p}{d}") % 50) + np.sin(d / 7) * 5
            promo = 1 if (d % 30) < 5 else 0
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "competitor_id": comp,
                "product_id": p,
                "price": round(price, 2),
                "discount": 0 if promo == 0 else round(10 + np.random.rand() * 15, 2),
                "region": "IN",
                "promo_flag": promo,
                "other_signal_1": np.random.rand()
            })

df_comp = pd.DataFrame(rows)
df_comp.to_csv("Sample/data/competitor_history.csv", index=False)
print("Created Sample/data/competitor_history.csv with", len(df_comp), "rows")

# --- review.csv ---
rev_rows = []
for i in range(1000):
    date = start + timedelta(days=np.random.randint(0, 300))
    prod = np.random.choice(products)
    rating = np.random.randint(1, 6)
    text = np.random.choice([
        "Great product, battery lasts long",
        "Poor build quality",
        "Good value for money",
        "Terrible customer service",
        "Satisfied, but camera could be better"
    ])
    rev_rows.append({
        "date": date.strftime("%Y-%m-%d"),
        "product_id": prod,
        "source": "amazon",
        "review_text": text,
        "rating": rating,
        "user_id": f"user_{i}"
    })

df_rev = pd.DataFrame(rev_rows)
df_rev.to_csv("Sample/data/review.csv", index=False)
print("Created Sample/data/review.csv with", len(df_rev), "rows")
