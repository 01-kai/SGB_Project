# check_convergence.py

import pandas as pd


def check_exp1_convergence(path="outputs/raw/exp1_summary_by_seed.csv"):
    df = pd.read_csv(path)

    seed_limits = [5, 10, 15, 20, 25, 30]

    rows = []

    for limit in seed_limits:
        subset = df[df["seed"] < limit]

        for framework, group in subset.groupby("framework"):
            rows.append({
                "seeds_used": limit,
                "framework": framework,
                "mean_E_GMI": group["steady_E_GMI"].mean(),
                "mean_success_rate": group["success_rate"].mean(),
                "mean_D1": group["D1"].mean(),
                "mean_D4": group["D4"].mean(),
            })

    result = pd.DataFrame(rows)

    print(result)

    result.to_csv(
        "outputs/tables/exp1_convergence_check.csv",
        index=False
    )

    print("\nSaved: outputs/tables/exp1_convergence_check.csv")


if __name__ == "__main__":
    check_exp1_convergence()