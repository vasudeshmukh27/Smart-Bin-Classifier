"""
Exploratory Data Analysis for 10,000 Image Dataset
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("AMAZON BIN IMAGE DATASET (10K) - EDA")
print("="*70)

BASE_PATH = Path("../data/raw")
IMG_PATH = BASE_PATH / "bin-images"
META_PATH = BASE_PATH / "metadata"
OUTPUT_PATH = Path("../outputs/visualizations/eda_10k")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Load metadata
print("\n[1/6] Loading 10,000 metadata files...")
metadata_list = []

for json_file in sorted(META_PATH.glob("*.json")):
    with open(json_file, 'r') as f:
        data = json.load(f)
        data['image_id'] = json_file.stem
        metadata_list.append(data)

print(f"✓ Loaded {len(metadata_list):,} metadata files")

# Extract statistics
print("\n[2/6] Extracting dataset statistics...")
total_quantities = []
num_unique_asins = []
all_asins = []

for meta in metadata_list:
    expected_qty = meta.get('EXPECTED_QUANTITY', 0)
    total_quantities.append(expected_qty)
    
    bin_data = meta.get('BIN_FCSKU_DATA', {})
    num_unique_asins.append(len(bin_data))
    
    for asin in bin_data.keys():
        all_asins.append(asin)

asin_counter = Counter(all_asins)

print("\n" + "="*70)
print("DATASET STATISTICS (10K Images)")
print("="*70)
print(f"Total bins: {len(metadata_list):,}")
print(f"Total unique ASINs: {len(asin_counter):,}")
print(f"\nQuantity per bin:")
print(f"  Mean: {np.mean(total_quantities):.2f}")
print(f"  Median: {np.median(total_quantities):.0f}")
print(f"  Min: {np.min(total_quantities)}")
print(f"  Max: {np.max(total_quantities)}")
print(f"\nUnique products per bin:")
print(f"  Mean: {np.mean(num_unique_asins):.2f}")
print(f"  Median: {np.median(num_unique_asins):.0f}")
print(f"  Min: {np.min(num_unique_asins)}")
print(f"  Max: {np.max(num_unique_asins)}")

# Visualizations
print("\n[3/6] Creating visualizations...")

# Quantity distributions
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(total_quantities, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Total Items per Bin')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Distribution of Item Quantities (10K Dataset)\nMean: {np.mean(total_quantities):.1f}')
axes[0].grid(True, alpha=0.3)

axes[1].hist(num_unique_asins, bins=range(1, max(num_unique_asins)+2), 
             edgecolor='black', alpha=0.7, color='coral')
axes[1].set_xlabel('Unique Products per Bin')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Distribution of Unique Products (10K Dataset)\nMean: {np.mean(num_unique_asins):.1f}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'quantity_distributions_10k.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: quantity_distributions_10k.png")
plt.close()

# Top ASINs
fig, ax = plt.subplots(figsize=(12, 8))
top_asins = asin_counter.most_common(30)
asins, counts = zip(*top_asins)
ax.barh(range(len(asins)), counts, color='teal')
ax.set_yticks(range(len(asins)))
ax.set_yticklabels([a[:12] + '...' for a in asins])
ax.set_xlabel('Frequency')
ax.set_title('Top 30 Most Common ASINs (10K Dataset)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'top_asins_10k.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: top_asins_10k.png")
plt.close()

print("\n[4/6] Comparing 999 vs 10K datasets...")
comparison_data = {
    'Metric': ['Total Images', 'Unique ASINs', 'Avg Items/Bin', 'Max Items/Bin'],
    '999 Images': [999, 1756, 5.33, 48],
    '10K Images': [len(metadata_list), len(asin_counter), np.mean(total_quantities), np.max(total_quantities)]
}
comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save statistics
stats_df = pd.DataFrame({
    'total_quantity': total_quantities,
    'unique_products': num_unique_asins
})
stats_df.to_csv(OUTPUT_PATH / 'dataset_statistics_10k.csv', index=False)
print(f"\n✓ Statistics saved to {OUTPUT_PATH / 'dataset_statistics_10k.csv'}")

print("\n[5/6] Analyzing data distribution for training...")
# Create distribution report
report = f"""
{'='*70}
10K DATASET ANALYSIS REPORT
{'='*70}

Dataset Size: {len(metadata_list):,} images
Unique Products: {len(asin_counter):,} ASINs

Quantity Statistics:
  Mean: {np.mean(total_quantities):.2f} items/bin
  Std Dev: {np.std(total_quantities):.2f}
  Range: {np.min(total_quantities)} - {np.max(total_quantities)}

Product Diversity:
  Mean unique products: {np.mean(num_unique_asins):.2f}
  Products appearing >100 times: {sum(1 for c in asin_counter.values() if c > 100)}
  Products appearing once: {sum(1 for c in asin_counter.values() if c == 1)}

Training Recommendations:
  - With 10K images, expect significantly better performance
  - Recommended split: 8000 train / 1000 val / 1000 test
  - Expected improvement over 999: 30-50% better accuracy
  - Can now train deeper models without overfitting

{'='*70}
"""

with open(OUTPUT_PATH / 'eda_report_10k.txt', 'w') as f:
    f.write(report)

print(report)
print(f"✓ Full report saved to {OUTPUT_PATH / 'eda_report_10k.txt'}")

print("\n[6/6] EDA Complete!")
print("="*70)
print("NEXT STEP: Train improved models with 10K dataset")
print("Expected improvement: 30-50% better performance than 999-image model")
print("="*70)
