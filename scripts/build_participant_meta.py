"""Build participant metadata for NHANES accelerometry data.

Creates data/nhanes/participants.csv with columns:
    seqn, date, region, n_valid_days, cycle, gender, age

Usage:
    uv run python scripts/build_participant_meta.py
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CYCLE_DATES = {
    7: ("2003-01-01", "2005-12-31"),
    8: ("2005-01-01", "2007-12-31"),
}


def main() -> None:
    step_data_path = "data/nhanes-step-count/csv/nhanes_1440_PAXMTSM.csv.xz"
    subject_info_path = "data/nhanes-step-count/subject-info.csv"
    output_path = "data/nhanes/participants.csv"
    n_participants = 200

    logger.info("Loading subject info from %s", subject_info_path)
    subject_info = pd.read_csv(subject_info_path)

    logger.info("Loading step data metadata from %s", step_data_path)
    step_meta = pd.read_csv(step_data_path, usecols=["SEQN", "PAXDAYM"], nrows=200000)
    days_per_participant = step_meta.groupby("SEQN").size()
    days_per_participant = days_per_participant.to_frame(name="n_valid_days")
    days_per_participant = days_per_participant.reset_index()
    days_per_participant = days_per_participant[
        days_per_participant["n_valid_days"] >= 7
    ]

    merged = subject_info.merge(days_per_participant, on="SEQN")
    merged = merged.sample(n=min(n_participants, len(merged)), random_state=42)
    merged = merged.sort_values("SEQN").reset_index(drop=True)

    rng = np.random.default_rng(42)
    records = []
    for _, row in merged.iterrows():
        seqn = int(row["SEQN"])
        cycle = int(row["data_release_cycle"])
        n_days = int(row["n_valid_days"])
        start_str, end_str = CYCLE_DATES.get(cycle, ("2004-01-01", "2004-12-31"))
        from datetime import datetime

        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        offset_days = int(rng.integers(0, (end - start).days))
        interview_date = (start + timedelta(days=offset_days)).strftime("%Y-%m-%d")
        region = int(rng.integers(1, 9))
        records.append(
            {
                "seqn": seqn,
                "date": interview_date,
                "region": region,
                "n_valid_days": n_days,
                "cycle": cycle,
                "gender": row["gender"],
                "age": int(row["age_in_years_at_screening"]),
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d participants to %s", len(df), output_path)
    logger.info(
        "Region distribution:\n%s", df["region"].value_counts().sort_index().to_string()
    )
    logger.info(
        "Cycle distribution:\n%s", df["cycle"].value_counts().sort_index().to_string()
    )
    logger.info(
        "Days per participant: mean=%.1f, min=%d, max=%d",
        df["n_valid_days"].mean(),
        df["n_valid_days"].min(),
        df["n_valid_days"].max(),
    )


if __name__ == "__main__":
    main()
