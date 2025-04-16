import pandas as pd

def load_team_averages(data_path="data/PL 2022_2023.csv", extra_path="data/PL 2023_2024.csv"):
    df1 = pd.read_csv(data_path)
    df2 = pd.read_csv(extra_path)
    df = pd.concat([df1, df2], ignore_index=True)

    teams = set(df["HomeTeam"].dropna().unique()).union(df["AwayTeam"].dropna().unique())

    team_stats = {}

    for team in teams:
        team_data = df[
            (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
        ]

        rows = []
        for _, row in team_data.iterrows():
            if row["HomeTeam"] == team:
                prefix = "H"
                opp_prefix = "A"
            else:
                prefix = "A"
                opp_prefix = "H"

            row_stats = {
                "HS": row[f"{prefix}S"],
                "AS": row[f"{opp_prefix}S"],
                "HST": row[f"{prefix}ST"],
                "AST": row[f"{opp_prefix}ST"],
                "HC": row[f"{prefix}C"],
                "AC": row[f"{opp_prefix}C"],
                "HF": row[f"{prefix}F"],
                "AF": row[f"{opp_prefix}F"],
                "HY": row[f"{prefix}Y"],
                "AY": row[f"{opp_prefix}Y"],
                "HR": row[f"{prefix}R"],
                "AR": row[f"{opp_prefix}R"]
            }
            rows.append(row_stats)

        if rows:
            avg_stats = pd.DataFrame(rows).mean().to_dict()
            team_stats[team] = avg_stats

    return team_stats

if __name__ == "__main__":
    stats = load_team_averages()
    for team, vals in list(stats.items())[:5]:
        print(f"\n{team}:")
        print(pd.Series(vals).round(2))
