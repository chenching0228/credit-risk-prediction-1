import os
import pandas as pd

print("=" * 50)
print("DATA VERIFICATION REPORT")
print("=" * 50)

# File paths
accepted_path = "data/raw/accepted_2007_to_2018Q4.csv"
rejected_path = "data/raw/rejected_2007_to_2018Q4.csv"

# Check files exist
files = {
    "Accepted loans": accepted_path,
    "Rejected loans": rejected_path
}

for name, path in files.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"\n✓ {name}: Found")
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"\n✗ {name}: NOT FOUND")
        print(f"  Expected location: {path}")

# Quick data load test
print("\n" + "=" * 50)
print("LOADING DATA SAMPLE...")
print("=" * 50)

try:
    # Load only first 1000 rows to test
    df_sample = pd.read_csv(accepted_path, nrows=1000)
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Sample rows: {len(df_sample)}")
    print(f"  Columns: {len(df_sample.columns)}")
    print(f"\n  First few columns: {', '.join(df_sample.columns[:5])}")
    
except Exception as e:
    print(f"\n✗ Error loading data: {str(e)}")

print("\n" + "=" * 50)
print("Verification complete!")
print("=" * 50)