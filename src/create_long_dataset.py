"""
Create a long-format dataset from a Qualtrics wide-format CSV.

This script:
1. Reads the raw Qualtrics CSV from data/raw/.
2. Removes the two Qualtrics metadata rows.
3. Keeps only completed and consented participants.
4. Converts participant-by-item columns into one row per participant x item.
5. Saves the cleaned long-format dataset and descriptive tables.

Raw participant data should never be committed to Git. Keep the raw CSV in
data/raw/, which is ignored by .gitignore.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import pandas as pd


# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

LONG_DATASET_PATH = PROCESSED_DATA_DIR / "long_dataset.csv"
DEFAULT_RAW_CSV_NAME = "results.csv"


# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------

ITEM_IDS = [1, 2, 3, 4, 5, 6, 8, 9]
EXPECTED_ITEMS_PER_PARTICIPANT = len(ITEM_IDS)

CONDITION_LABELS = {
    "CE": "correct AI answer + explanation/reasoning",
    "CN": "correct AI answer + no explanation/reasoning",
    "IE": "incorrect AI answer + explanation/reasoning",
    "IN": "incorrect AI answer + no explanation/reasoning",
}

# The embedded-data condition columns may use descriptive labels, while the
# Qualtrics question columns use short codes like I01_CE_A1.
CONDITION_CODE_MAP = {
    "CE": "CE",
    "CN": "CN",
    "IE": "IE",
    "IN": "IN",
    "correct_explanation": "CE",
    "correct_no_explanation": "CN",
    "incorrect_explanation": "IE",
    "incorrect_no_explanation": "IN",
}

LIKERT_MAP = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neither agree nor disagree": 3,
    "Agree": 4,
    "Strongly agree": 5,
}

CONFIDENCE_MAP = {
    "Not confident at all": 1,
    "Slightly confident": 2,
    "Moderately confident": 3,
    "Confident": 4,
    "Very confident": 5,
    "Very Confident": 5,
}

CORRECT_ANSWER_KEY = {
    1: "India",
    2: "45",
    3: "Canberra",
    4: "Both at the same time",
    5: "Mercury",
    6: "Cats are animals",
    8: "Hesitant",
    9: "2",
}

INCORRECT_AI_ANSWER_KEY = {
    1: "China",
    2: "40",
    3: "Sydney",
    4: "A rock",
    5: "Venus",
    6: "Some cats are black",
    8: "Willing",
    9: "3",
}


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def find_raw_csv(raw_csv_path: str | Path | None = None) -> Path:
    """Return the raw CSV path, or find one CSV in data/raw/."""
    if raw_csv_path is not None:
        path = Path(raw_csv_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Raw CSV file does not exist: {path}")
        return path

    default_csv_path = RAW_DATA_DIR / DEFAULT_RAW_CSV_NAME
    if default_csv_path.exists():
        return default_csv_path

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No raw CSV found. Put your Qualtrics CSV in data/raw/ or pass "
            "a path to main(raw_csv_path=...)."
        )

    if len(csv_files) > 1:
        file_list = "\n".join(f"- {path.name}" for path in csv_files)
        raise ValueError(
            "More than one CSV was found in data/raw/. Please pass the exact "
            f"file path to main(raw_csv_path=...). Found:\n{file_list}"
        )

    return csv_files[0]


def normalize_answer(value: object) -> str | None:
    """Convert answers to comparable strings while keeping missing values."""
    if pd.isna(value):
        return None
    return str(value).strip().lower()


def answers_match(answer: object, correct_answer: object) -> int | Any:
    """Return 1 if two answers match, 0 if they differ, and NA if unavailable."""
    if pd.isna(answer) or correct_answer is None or pd.isna(correct_answer):
        return pd.NA
    return int(normalize_answer(answer) == normalize_answer(correct_answer))


def map_yes_no(value: object, yes_value: str, no_value: str, column_name: str) -> int | Any:
    """Map text values such as yes/no or correct/incorrect to 1/0."""
    if pd.isna(value):
        return pd.NA

    cleaned = str(value).strip().lower()
    if cleaned == yes_value:
        return 1
    if cleaned == no_value:
        return 0

    warnings.warn(
        f"Unexpected value in {column_name}: {value!r}. Returning missing value.",
        stacklevel=2,
    )
    return pd.NA


def require_columns(data: pd.DataFrame, required_columns: list[str], context: str) -> None:
    """Raise a clear error if any required columns are missing."""
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        missing = "\n".join(f"- {column}" for column in missing_columns)
        raise ValueError(f"Missing required columns for {context}:\n{missing}")


def convert_likert(series: pd.Series) -> pd.Series:
    """Convert agreement response text to numeric Likert values."""
    return series.map(LIKERT_MAP)


def convert_confidence(series: pd.Series) -> pd.Series:
    """Convert confidence response text to numeric values."""
    return series.map(CONFIDENCE_MAP)


def get_actual_ai_answer(item_id: int, ai_correctness: object) -> str | None:
    """Return the AI answer shown to the participant for this item/condition."""
    if pd.isna(ai_correctness):
        return None
    if ai_correctness == 1:
        return CORRECT_ANSWER_KEY[item_id]
    return INCORRECT_AI_ANSWER_KEY[item_id]


def cronbach_alpha(data: pd.DataFrame, columns: list[str]) -> float:
    """
    Calculate Cronbach's alpha for a set of scale items.

    Alpha is most useful when a scale has multiple items. Here each scale has
    two items, so this is a simple reliability check for those paired items.
    """
    scale_data = data[columns].dropna()

    if scale_data.shape[0] < 2:
        return float("nan")

    item_variances = scale_data.var(axis=0, ddof=1)
    total_score_variance = scale_data.sum(axis=1).var(ddof=1)
    number_of_items = len(columns)

    if total_score_variance == 0:
        return float("nan")

    alpha = (
        number_of_items
        / (number_of_items - 1)
        * (1 - item_variances.sum() / total_score_variance)
    )
    return float(alpha)


# ---------------------------------------------------------------------------
# Main data-processing functions
# ---------------------------------------------------------------------------

def read_and_filter_raw_data(raw_csv_path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw Qualtrics CSV, remove metadata rows, and filter responses."""
    csv_path = find_raw_csv(raw_csv_path)
    raw_data = pd.read_csv(csv_path)
    print(f"Raw CSV path: {csv_path}")
    print(f"Raw shape, including metadata rows: {raw_data.shape}")

    data = raw_data.iloc[2:].copy()
    print(f"Shape after removing 2 metadata rows: {data.shape}")

    require_columns(data, ["Finished", "consent agree", "SurveyVersion"], "participant filtering")

    completed_text = data["Finished"].astype(str).str.strip().str.lower()
    consent_text = data["consent agree"].astype(str).str.strip().str.lower()

    completed = completed_text.isin(["true", "1", "yes"])
    consented = consent_text.isin(["yes", "true", "1", "agree"]) | consent_text.str.startswith("yes,")

    print(f"Completed responses: {completed.sum()}")
    print(f"Consented participants: {(completed & consented).sum()}")

    filtered_data = data.loc[completed & consented].copy()
    if filtered_data.empty:
        consent_values = data["consent agree"].value_counts(dropna=False).to_string()
        raise ValueError(
            "No participants remained after filtering for completed and consented responses. "
            "Check the accepted consent labels in read_and_filter_raw_data().\n\n"
            f"Observed consent values:\n{consent_values}"
        )

    filtered_data = filtered_data.reset_index(drop=True)
    filtered_data["participant_id"] = range(1, len(filtered_data) + 1)

    return filtered_data


def expected_item_columns(item_id: int, condition_code: str) -> dict[str, str]:
    """Return the expected Qualtrics question columns for an item condition."""
    prefix = f"I{item_id:02d}_{condition_code}"
    return {
        "initial_answer": f"{prefix}_A1",
        "initial_confidence": f"{prefix}_C1",
        "final_answer": f"{prefix}_A2",
        "final_confidence": f"{prefix}_C2",
        "trust_item_trust": f"{prefix}_E_1",
        "trust_item_reliable": f"{prefix}_E_2",
        "reliance_item_influenced": f"{prefix}_E_3",
        "reliance_item_rely_real": f"{prefix}_E_4",
    }


def build_long_dataset(filtered_data: pd.DataFrame) -> pd.DataFrame:
    """Build one row per participant x item."""
    base_required_columns = ["participant_id", "SurveyVersion"]
    embedded_columns = []

    for item_id in ITEM_IDS:
        embedded_columns.extend(
            [
                f"Item{item_id}_condition",
                f"Item{item_id}_accuracy",
                f"Item{item_id}_explanation",
            ]
        )

    require_columns(filtered_data, base_required_columns + embedded_columns, "long dataset setup")

    long_rows = []

    for _, participant in filtered_data.iterrows():
        for item_order, item_id in enumerate(ITEM_IDS, start=1):
            condition_column = f"Item{item_id}_condition"
            accuracy_column = f"Item{item_id}_accuracy"
            explanation_column = f"Item{item_id}_explanation"

            condition_value = str(participant[condition_column]).strip()
            condition_code = CONDITION_CODE_MAP.get(condition_value, condition_value)
            if condition_code not in CONDITION_LABELS:
                raise ValueError(
                    f"Unexpected condition value {condition_value!r} for participant "
                    f"{participant['participant_id']} item {item_id}."
                )

            item_columns = expected_item_columns(item_id, condition_code)
            require_columns(
                filtered_data,
                list(item_columns.values()),
                f"participant item columns for item {item_id} condition {condition_code}",
            )

            initial_confidence = CONFIDENCE_MAP.get(str(participant[item_columns["initial_confidence"]]).strip())
            final_confidence = CONFIDENCE_MAP.get(str(participant[item_columns["final_confidence"]]).strip())

            trust_item_trust = LIKERT_MAP.get(str(participant[item_columns["trust_item_trust"]]).strip())
            trust_item_reliable = LIKERT_MAP.get(str(participant[item_columns["trust_item_reliable"]]).strip())
            reliance_item_influenced = LIKERT_MAP.get(str(participant[item_columns["reliance_item_influenced"]]).strip())
            reliance_item_rely_real = LIKERT_MAP.get(str(participant[item_columns["reliance_item_rely_real"]]).strip())

            ai_correctness = map_yes_no(
                participant[accuracy_column],
                yes_value="correct",
                no_value="incorrect",
                column_name=accuracy_column,
            )
            reasoning = map_yes_no(
                participant[explanation_column],
                yes_value="yes",
                no_value="no",
                column_name=explanation_column,
            )

            initial_answer = participant[item_columns["initial_answer"]]
            final_answer = participant[item_columns["final_answer"]]
            actual_ai_answer = get_actual_ai_answer(item_id, ai_correctness)

            row = {
                "participant_id": participant["participant_id"],
                "survey_version": participant["SurveyVersion"],
                "item_id": item_id,
                "item_order": item_order,
                "condition_code": condition_code,
                "condition_label": CONDITION_LABELS[condition_code],
                "reasoning": reasoning,
                "ai_correctness": ai_correctness,
                "initial_answer": initial_answer,
                "final_answer": final_answer,
                "initial_confidence": initial_confidence,
                "final_confidence": final_confidence,
                "confidence_change": (
                    final_confidence - initial_confidence
                    if initial_confidence is not None and final_confidence is not None
                    else pd.NA
                ),
                "initial_correct": answers_match(initial_answer, CORRECT_ANSWER_KEY[item_id]),
                "final_correct": answers_match(final_answer, CORRECT_ANSWER_KEY[item_id]),
                "changed_answer": answers_match(final_answer, initial_answer) if not pd.isna(initial_answer) else pd.NA,
                "actual_ai_answer": actual_ai_answer,
                "changed_to_ai_answer": answers_match(final_answer, actual_ai_answer),
                "trust_item_trust": trust_item_trust,
                "trust_item_reliable": trust_item_reliable,
                "trust_score": pd.Series([trust_item_trust, trust_item_reliable], dtype="float").mean(),
                "reliance_item_influenced": reliance_item_influenced,
                "reliance_item_rely_real": reliance_item_rely_real,
                "reliance_score": pd.Series([reliance_item_influenced, reliance_item_rely_real], dtype="float").mean(),
            }

            # changed_answer should be 1 when answers differ, not when they match.
            if not pd.isna(row["changed_answer"]):
                row["changed_answer"] = 1 - row["changed_answer"]

            long_rows.append(row)

    long_data = pd.DataFrame(long_rows)

    numeric_columns = [
        "reasoning",
        "ai_correctness",
        "initial_confidence",
        "final_confidence",
        "confidence_change",
        "initial_correct",
        "final_correct",
        "changed_answer",
        "changed_to_ai_answer",
        "trust_item_trust",
        "trust_item_reliable",
        "trust_score",
        "reliance_item_influenced",
        "reliance_item_rely_real",
        "reliance_score",
    ]

    for column in numeric_columns:
        long_data[column] = pd.to_numeric(long_data[column], errors="coerce")

    return long_data


def validate_long_dataset(long_data: pd.DataFrame, participant_count: int) -> None:
    """Print validation checks for the long-format dataset."""
    expected_rows = participant_count * EXPECTED_ITEMS_PER_PARTICIPANT
    actual_rows = len(long_data)

    print(f"Expected long-format rows: {expected_rows}")
    print(f"Actual long-format rows: {actual_rows}")

    if actual_rows != expected_rows:
        warnings.warn(
            f"Long-format row count is {actual_rows}, but expected {expected_rows} "
            f"({participant_count} participants x {EXPECTED_ITEMS_PER_PARTICIPANT} items).",
            stacklevel=2,
        )

    rows_per_participant = long_data.groupby("participant_id").size()
    participants_with_wrong_item_count = rows_per_participant[
        rows_per_participant != EXPECTED_ITEMS_PER_PARTICIPANT
    ]

    if participants_with_wrong_item_count.empty:
        print("Each participant has exactly 8 item rows.")
    else:
        print("Participants without exactly 8 item rows:")
        print(participants_with_wrong_item_count)

    for score_column in ["trust_score", "reliance_score"]:
        missing_share = long_data[score_column].isna().mean()
        print(f"Missing share for {score_column}: {missing_share:.1%}")
        if missing_share > 0.25:
            warnings.warn(
                f"{score_column} is more than 25% missing. Check response labels and column names.",
                stacklevel=2,
            )

    print("Condition counts:")
    print(long_data["condition_label"].value_counts(dropna=False))

    missing_incorrect_answers = [
        item_id for item_id, answer in INCORRECT_AI_ANSWER_KEY.items() if answer is None
    ]
    if missing_incorrect_answers:
        warnings.warn(
            "Incorrect AI answer placeholders are still empty for item(s): "
            f"{missing_incorrect_answers}. Fill INCORRECT_AI_ANSWER_KEY to compute "
            "changed_to_ai_answer for incorrect-AI conditions.",
            stacklevel=2,
        )


def create_descriptive_tables(long_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create descriptive summary tables."""
    condition_counts = (
        long_data.groupby("condition_label", dropna=False)
        .agg(
            n_observations=("participant_id", "size"),
            n_participants=("participant_id", "nunique"),
        )
        .reset_index()
    )

    summary_columns = {
        "n_observations": ("participant_id", "size"),
        "trust_score_mean": ("trust_score", "mean"),
        "trust_score_sd": ("trust_score", "std"),
        "reliance_score_mean": ("reliance_score", "mean"),
        "reliance_score_sd": ("reliance_score", "std"),
        "changed_answer_mean": ("changed_answer", "mean"),
        "changed_to_ai_answer_mean": ("changed_to_ai_answer", "mean"),
        "initial_correct_mean": ("initial_correct", "mean"),
        "final_correct_mean": ("final_correct", "mean"),
        "confidence_change_mean": ("confidence_change", "mean"),
    }

    descriptive_by_condition = (
        long_data.groupby(["ai_correctness", "reasoning", "condition_label"], dropna=False)
        .agg(**summary_columns)
        .reset_index()
    )

    descriptive_by_item_condition = (
        long_data.groupby(["item_id", "condition_label"], dropna=False)
        .agg(**summary_columns)
        .reset_index()
    )

    return {
        "condition_counts": condition_counts,
        "descriptive_by_condition": descriptive_by_condition,
        "descriptive_by_item_condition": descriptive_by_item_condition,
    }


def save_outputs(long_data: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    """Save the long dataset and descriptive tables as CSV files."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    long_data.to_csv(LONG_DATASET_PATH, index=False)
    print(f"Saved long dataset to: {LONG_DATASET_PATH}")

    for table_name, table in tables.items():
        output_path = TABLES_DIR / f"{table_name}.csv"
        table.to_csv(output_path, index=False)
        print(f"Saved {table_name} to: {output_path}")


def main(raw_csv_path: str | Path | None = None) -> pd.DataFrame:
    """Run the full cleaning pipeline and return the long dataset."""
    filtered_data = read_and_filter_raw_data(raw_csv_path)
    participant_count = filtered_data["participant_id"].nunique()

    long_data = build_long_dataset(filtered_data)
    validate_long_dataset(long_data, participant_count)

    trust_alpha = cronbach_alpha(long_data, ["trust_item_trust", "trust_item_reliable"])
    reliance_alpha = cronbach_alpha(
        long_data,
        ["reliance_item_influenced", "reliance_item_rely_real"],
    )

    print(f"Cronbach's alpha, trust scale: {trust_alpha:.3f}")
    print(f"Cronbach's alpha, reliance scale: {reliance_alpha:.3f}")

    tables = create_descriptive_tables(long_data)
    save_outputs(long_data, tables)

    return long_data


if __name__ == "__main__":
    main()
