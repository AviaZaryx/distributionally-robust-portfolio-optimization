import pandas as pd
import ast
import os

# --- CONFIG ---
FILE_PATH = r'D:\Downloads\pythonProject1\time_scale_size.csv'
BACKUP_PATH = r'D:\Downloads\pythonProject1\time_scale_size_decimal_backup.csv'


def format_tuple_string(val):
    """
    Takes a string like "(0.077, 4.1e-05)" and
    returns "(0.077000, 0.000041)"
    """
    if not isinstance(val, str) or not val.startswith('('):
        return val

    try:
        # Convert string "(1.2, 3.4e-05)" into a Python tuple
        # Note: ast.literal_eval safely parses the string into a tuple of numbers
        data_tuple = ast.literal_eval(val)

        # Format each number in the tuple to 6 decimal places as a string
        # Change ":.6f" to ":.8f" if you need more precision
        formatted_elements = [f"{float(x):.6f}" for x in data_tuple]

        # Join them back into the tuple string format
        return "(" + ", ".join(formatted_elements) + ")"

    except (ValueError, SyntaxError):
        # If it's not a valid tuple or contains 'nan', return as is
        return val


def fix_scientific_notation():
    if not os.path.exists(FILE_PATH):
        print(f"Error: Could not find file at {FILE_PATH}")
        return

    print("Reading CSV...")
    df = pd.read_csv(FILE_PATH)

    # Save backup
    df.to_csv(BACKUP_PATH, index=False)
    print(f"Backup created at: {BACKUP_PATH}")

    print("Converting scientific notation to decimals...")
    # Apply formatting to all columns except 'solver'
    for col in df.columns:
        if col != 'solver':
            df[col] = df[col].astype(str).map(format_tuple_string)

    print("Saving cleaned CSV...")
    df.to_csv(FILE_PATH, index=False)

    print("-" * 30)
    print("SUCCESS: Scientific notation converted to decimals.")
    print("Example: 4.1e-05 is now 0.000041")


if __name__ == "__main__":
    fix_scientific_notation()