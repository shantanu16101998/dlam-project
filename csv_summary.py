import pandas as pd
import os
import json

FILES = [
    r"./data/input/train.csv",
    r"./data/input/forecast_index_validation.csv",
]

def summarize_csv(path):
    print("\n" + "=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    if not os.path.exists(path):
        print("FILE NOT FOUND")
        return

    # Read only enough to infer structure while still getting useful statistics
    df = pd.read_csv(path)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n--- Columns ---")
    for i, col in enumerate(df.columns):
        print(f"{i:3d}. {col!r:40s} dtype={df[col].dtype}")

    print("\n--- Missing values ---")
    missing = df.isna().sum()
    missing_pct = 100 * missing / len(df)

    for col in df.columns:
        if missing[col] > 0:
            print(
                f"{col!r:40s} "
                f"{missing[col]:,} missing "
                f"({missing_pct[col]:.2f}%)"
            )

    if missing.sum() == 0:
        print("No missing values.")

    print("\n--- First 5 rows ---")
    print(df.head(5).to_string(index=False))

    print("\n--- Last 5 rows ---")
    print(df.tail(5).to_string(index=False))

    print("\n--- Numeric columns ---")
    numeric = df.select_dtypes(include="number")

    if len(numeric.columns):
        print(
            numeric.describe()
            .T[
                ["count", "mean", "std", "min",
                 "25%", "50%", "75%", "max"]
            ]
            .to_string()
        )
    else:
        print("No numeric columns.")

    print("\n--- Non-numeric columns ---")
    non_numeric = df.select_dtypes(exclude="number")

    for col in non_numeric.columns:
        n_unique = df[col].nunique(dropna=False)

        print(f"\n{col!r}")
        print(f"  unique values: {n_unique:,}")

        # Show values only when the cardinality is reasonably small
        if n_unique <= 30:
            print("  values:")
            print(
                df[col]
                .value_counts(dropna=False)
                .head(30)
                .to_string()
            )
        else:
            print("  sample values:")
            print(df[col].drop_duplicates().head(10).tolist())

    print("\n--- Possible datetime columns ---")

    for col in df.columns:
        # Only test object/string columns
        if df[col].dtype == "object":
            converted = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            success_rate = converted.notna().mean()

            if success_rate >= 0.8:
                print(
                    f"{col!r}: "
                    f"{success_rate * 100:.1f}% successfully parsed as datetime"
                )

    print("\n--- Duplicate rows ---")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n--- Memory usage ---")
    print(
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )


for file in FILES:
    summarize_csv(file)

print("\n" + "=" * 80)
print("SUMMARY COMPLETE")
print("=" * 80)