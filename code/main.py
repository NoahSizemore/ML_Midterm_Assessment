from table_clean import clean_table, save_cleaned_table
from pathlib import Path
file_path = Path(__file__).parent / "train.xlsx"


def main():
    file_path = Path(__file__).parent / "train.xlsx"
    output_path = Path(__file__).parent / "cleaned_output.xlsx"
    cleaned_df = clean_table(file_path)
    save_cleaned_table(cleaned_df, output_path)

if __name__ == "__main__":
    main()