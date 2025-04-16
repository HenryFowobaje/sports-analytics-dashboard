import pandas as pd
import glob


def load_and_process_data():
    files = glob.glob("data/PL *.csv")  # Load all files matching pattern
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Clean: drop rows with missing values in key columns
    cols_needed = ["FTR", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]
    df = df.dropna(subset=cols_needed)

    # Map match result: H=1, D=0, A=-1
    df["match_result"] = df["FTR"].map({"H": 1, "D": 0, "A": -1})

    # Select pre-match features
    features = [
        "HS",  # Home Shots
        "AS",  # Away Shots
        "HST", # Home Shots on Target
        "AST", # Away Shots on Target
        "HC",  # Home Corners
        "AC",  # Away Corners
        "HF",  # Home Fouls
        "AF",  # Away Fouls
        "HY",  # Home Yellow Cards
        "AY",  # Away Yellow Cards
        "HR",  # Home Red Cards
        "AR",  # Away Red Cards
    ]

    df = df[features + ["match_result"]]
    return df

if __name__ == "__main__":
    df = load_and_process_data()
    print(df.head())
