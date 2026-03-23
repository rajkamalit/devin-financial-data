"""
Fraud Risk Scorer

Analyzes financial transactions and assigns risk scores (0-100) based on
multiple risk signals including transaction amount, type, account behavior,
and transaction patterns.

Risk Levels:
    LOW:    score < 40
    MEDIUM: 40 <= score <= 70
    HIGH:   score > 70
"""

import argparse
import os
from datetime import datetime, timezone

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HIGH_AMOUNT_THRESHOLD = 10_000
FLAGGED_AMOUNT_THRESHOLD = 200_000
HIGH_RISK_TYPES = {"CASH_OUT", "TRANSFER"}

# Weight each risk signal contributes to the final score (out of 100)
WEIGHT_AMOUNT = 25
WEIGHT_TYPE = 20
WEIGHT_BALANCE_DRAIN = 15
WEIGHT_DEST_FREQUENCY = 10
WEIGHT_RAPID_TRANSACTIONS = 15
WEIGHT_BALANCE_MISMATCH = 15


def classify_risk(score: float) -> str:
    """Assign a risk level label based on the numeric score."""
    if score < 40:
        return "LOW"
    if score <= 70:
        return "MEDIUM"
    return "HIGH"


# ---------------------------------------------------------------------------
# Individual risk signal scorers
# ---------------------------------------------------------------------------

def score_amount(amount: float) -> float:
    """Score based on transaction amount.

    - Amounts above the flagged threshold (200k) get full points.
    - Amounts above the high threshold (10k) scale linearly.
    - Amounts below the threshold get minimal points.
    """
    if amount >= FLAGGED_AMOUNT_THRESHOLD:
        return WEIGHT_AMOUNT
    if amount >= HIGH_AMOUNT_THRESHOLD:
        ratio = (amount - HIGH_AMOUNT_THRESHOLD) / (
            FLAGGED_AMOUNT_THRESHOLD - HIGH_AMOUNT_THRESHOLD
        )
        return WEIGHT_AMOUNT * (0.4 + 0.6 * ratio)
    # Small amounts still get a tiny base score
    return WEIGHT_AMOUNT * 0.1 * (amount / HIGH_AMOUNT_THRESHOLD)


def score_type(txn_type: str) -> float:
    """Score based on transaction type. CASH_OUT and TRANSFER are riskier."""
    if txn_type in HIGH_RISK_TYPES:
        return WEIGHT_TYPE
    return 0.0


def score_balance_drain(
    old_balance: float, new_balance: float, amount: float
) -> float:
    """Score based on how much of the originator's balance is drained.

    Fully draining an account is a strong fraud signal.
    """
    if old_balance == 0:
        return WEIGHT_BALANCE_DRAIN * 0.5 if amount > 0 else 0.0
    drain_ratio = (old_balance - new_balance) / old_balance
    if drain_ratio >= 1.0:
        return WEIGHT_BALANCE_DRAIN
    if drain_ratio >= 0.8:
        return WEIGHT_BALANCE_DRAIN * 0.7
    if drain_ratio >= 0.5:
        return WEIGHT_BALANCE_DRAIN * 0.3
    return 0.0


def compute_dest_frequency_scores(df: pd.DataFrame) -> pd.Series:
    """Score based on how often a destination account appears.

    Destinations that appear only once (previously unseen) are riskier.
    """
    dest_counts = df["nameDest"].value_counts()
    total = len(df)

    def _score(dest: str) -> float:
        count = dest_counts.get(dest, 1)
        if count == 1:
            return WEIGHT_DEST_FREQUENCY
        # More appearances → less risky (destination is 'known')
        return WEIGHT_DEST_FREQUENCY * max(0, 1 - count / total)

    return df["nameDest"].apply(_score)


def compute_rapid_transaction_scores(df: pd.DataFrame) -> pd.Series:
    """Score based on rapid sequences of transactions from the same account.

    Multiple transactions within the same or consecutive time-steps
    from the same originator indicate potential fraud.
    """
    orig_step_counts = (
        df.groupby("nameOrig")["step"]
        .transform("count")
    )
    max_count = orig_step_counts.max() if orig_step_counts.max() > 1 else 1

    scores = (orig_step_counts - 1) / (max_count - 1) if max_count > 1 else 0.0
    return scores * WEIGHT_RAPID_TRANSACTIONS


def score_balance_mismatch(
    amount: float,
    old_balance_orig: float,
    new_balance_orig: float,
    old_balance_dest: float,
    new_balance_dest: float,
) -> float:
    """Score based on balance discrepancies.

    If the amount doesn't reconcile with balance changes (ignoring
    merchant destinations starting with 'M'), flag as suspicious.
    """
    # Check originator side
    expected_new_orig = old_balance_orig - amount
    orig_mismatch = abs(new_balance_orig - expected_new_orig) > 1.0

    # Destination side (skip merchants whose balances are always 0)
    if old_balance_dest == 0 and new_balance_dest == 0:
        dest_mismatch = False
    else:
        expected_new_dest = old_balance_dest + amount
        dest_mismatch = abs(new_balance_dest - expected_new_dest) > 1.0

    if orig_mismatch and dest_mismatch:
        return WEIGHT_BALANCE_MISMATCH
    if orig_mismatch or dest_mismatch:
        return WEIGHT_BALANCE_MISMATCH * 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Main scoring pipeline
# ---------------------------------------------------------------------------

def build_explanation(components: dict[str, float]) -> str:
    """Build a human-readable explanation from scored components."""
    parts = []
    for signal, value in components.items():
        if value > 0:
            parts.append(f"{signal}: {value:.1f}")
    return "; ".join(parts) if parts else "No significant risk signals"


def score_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Score every transaction in *df* and return a results DataFrame.

    Returns a DataFrame with columns:
        transaction_id, risk_score, risk_level, explanation
    """
    # Assign a transaction_id if not present
    df = df.copy()
    df["transaction_id"] = [f"TXN-{i+1:06d}" for i in range(len(df))]

    # Pre-compute vectorised signals
    dest_freq_scores = compute_dest_frequency_scores(df)
    rapid_scores = compute_rapid_transaction_scores(df)

    results = []
    for idx, row in df.iterrows():
        components: dict[str, float] = {}

        # 1. Amount risk
        components["high_amount"] = score_amount(row["amount"])

        # 2. Transaction type risk
        components["risky_type"] = score_type(row["type"])

        # 3. Balance drain
        components["balance_drain"] = score_balance_drain(
            row["oldbalanceOrg"], row["newbalanceOrig"], row["amount"]
        )

        # 4. Destination frequency (unseen accounts)
        components["unseen_dest"] = dest_freq_scores.loc[idx]

        # 5. Rapid transactions from same account
        components["rapid_txns"] = rapid_scores.loc[idx]

        # 6. Balance mismatch
        components["balance_mismatch"] = score_balance_mismatch(
            row["amount"],
            row["oldbalanceOrg"],
            row["newbalanceOrig"],
            row["oldbalanceDest"],
            row["newbalanceDest"],
        )

        total_score = min(100.0, sum(components.values()))
        total_score = round(total_score, 2)

        results.append(
            {
                "transaction_id": row["transaction_id"],
                "risk_score": total_score,
                "risk_level": classify_risk(total_score),
                "explanation": build_explanation(components),
            }
        )

    return pd.DataFrame(results)


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV dataset and perform basic validation."""
    required_columns = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    }

    df = pd.read_csv(path)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing))}"
        )
    return df


def generate_report(
    scored_df: pd.DataFrame, output_path: str | None = None
) -> str:
    """Generate a summary report and optionally save to a file."""
    total = len(scored_df)
    high = len(scored_df[scored_df["risk_level"] == "HIGH"])
    medium = len(scored_df[scored_df["risk_level"] == "MEDIUM"])
    low = len(scored_df[scored_df["risk_level"] == "LOW"])
    avg_score = scored_df["risk_score"].mean()

    report_lines = [
        "=" * 60,
        "         FRAUD RISK SCORING REPORT",
        f"         Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC",
        "=" * 60,
        "",
        f"Total Transactions Analyzed:  {total}",
        f"Average Risk Score:           {avg_score:.2f}",
        "",
        "Risk Distribution:",
        f"  HIGH   (score > 70):   {high:>5}  ({high/total*100:.1f}%)",
        f"  MEDIUM (40-70):        {medium:>5}  ({medium/total*100:.1f}%)",
        f"  LOW    (score < 40):   {low:>5}  ({low/total*100:.1f}%)",
        "",
        "-" * 60,
        "Top 10 Highest Risk Transactions:",
        "-" * 60,
    ]

    top10 = scored_df.nlargest(10, "risk_score")
    for _, row in top10.iterrows():
        report_lines.append(
            f"  {row['transaction_id']}  "
            f"Score: {row['risk_score']:>6.2f}  "
            f"Level: {row['risk_level']:<6}  "
            f"| {row['explanation']}"
        )

    report_lines.append("")
    report_lines.append("=" * 60)
    report = "\n".join(report_lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign fraud risk scores to financial transactions."
    )
    parser.add_argument(
        "--input",
        default=os.path.join("data", "sample_transactions.csv"),
        help="Path to the input CSV file (default: data/sample_transactions.csv)",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join("output", "transaction_risk_scores.csv"),
        help="Path to save scored CSV (default: output/transaction_risk_scores.csv)",
    )
    parser.add_argument(
        "--output-report",
        default=os.path.join("output", "risk_report.txt"),
        help="Path to save the summary report (default: output/risk_report.txt)",
    )

    args = parser.parse_args()

    # Load
    print(f"Loading dataset from: {args.input}")
    df = load_dataset(args.input)
    print(f"Loaded {len(df)} transactions.")

    # Score
    print("Computing risk scores …")
    scored_df = score_transactions(df)

    # Save scored output
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    scored_df.to_csv(args.output_csv, index=False)
    print(f"Scored transactions saved to: {args.output_csv}")

    # Generate and print report
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    report = generate_report(scored_df, output_path=args.output_report)
    print(f"Report saved to: {args.output_report}\n")
    print(report)


if __name__ == "__main__":
    main()
