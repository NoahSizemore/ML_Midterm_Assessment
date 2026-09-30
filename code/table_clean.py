import pandas as pd
import numpy as np

def clean_table(file_path):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    print("="*40)
    print("=== TABLE LOADED ===")
    print("="*40)
    print(f"Table loaded successfully. Shape: {df.shape}")
    print("="*40)   
    print(f"Head of the table:\n{df.head()}")

    # Clean rows with impossible values and change to NaN
    df.loc[df["Sleep Hours"] < 0, "Sleep Hours"] = np.nan
    df.loc[df["Previous Scores"] > 100, "Previous Scores"] = np.nan
    df.loc[df["Previous Scores"] < 0, "Previous Scores"] = np.nan
    df.loc[df["Hours Studied"] > 24, "Hours Studied"] = np.nan

    # Fill missing values with the median of each column
    for col in df.columns:
        df[col] = df[col].fillna(df[col].median())

    # Print the cleaned DataFrame to verify the changes
    print("="*40)
    print("=== VERIFICATION ===")
    print("="*40)
    print(f"Cleaned table. Shape: {df.shape}")
    print("="*40)   
    print(f"Head of the table:\n{df.head()}")
    print("="*40)   
    print(f"\ndf.isna().sum():\t{df.isna().sum()}")
    print(f"df.duplicated().sum():\t{df.duplicated().sum()}")
    print("="*40)
    print("=== END OF VERIFICATION ===")

    return df

def save_cleaned_table(df, output_path):
    df.to_excel(output_path, index=False)
    print(f"Cleaned table saved to {output_path}")