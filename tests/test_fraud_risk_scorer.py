"""Tests for fraud_risk_scorer.py"""

import os
import sys

import pandas as pd
import pytest

# Ensure the repo root is on the path so we can import the scorer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fraud_risk_scorer import (
    build_explanation,
    classify_risk,
    compute_dest_frequency_scores,
    compute_rapid_transaction_scores,
    generate_report,
    load_dataset,
    score_amount,
    score_balance_drain,
    score_balance_mismatch,
    score_transactions,
    score_type,
)


# ---------------------------------------------------------------------------
# 1. test_classify_risk
# ---------------------------------------------------------------------------

class TestClassifyRisk:
    def test_low_boundary_zero(self):
        assert classify_risk(0) == "LOW"

    def test_low_boundary_just_below(self):
        assert classify_risk(39.9) == "LOW"

    def test_medium_boundary_exact(self):
        assert classify_risk(40) == "MEDIUM"

    def test_medium_boundary_upper(self):
        assert classify_risk(70) == "MEDIUM"

    def test_high_boundary_just_above(self):
        assert classify_risk(70.1) == "HIGH"

    def test_high_boundary_max(self):
        assert classify_risk(100) == "HIGH"


# ---------------------------------------------------------------------------
# 2. test_score_amount
# ---------------------------------------------------------------------------

class TestScoreAmount:
    def test_at_flagged_threshold(self):
        assert score_amount(200_000) == 25

    def test_above_flagged_threshold(self):
        assert score_amount(500_000) == 25

    def test_between_thresholds(self):
        # 25 * (0.4 + 0.6 * (105000 - 10000) / (200000 - 10000))
        # = 25 * (0.4 + 0.6 * 0.5) = 17.5
        assert score_amount(105_000) == pytest.approx(17.5)

    def test_at_high_threshold(self):
        # 25 * 0.4 = 10.0
        assert score_amount(10_000) == pytest.approx(10.0)

    def test_below_high_threshold(self):
        # 25 * 0.1 * (5000 / 10000) = 1.25
        assert score_amount(5_000) == pytest.approx(1.25)

    def test_zero_amount(self):
        assert score_amount(0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. test_score_type
# ---------------------------------------------------------------------------

class TestScoreType:
    def test_cash_out(self):
        assert score_type("CASH_OUT") == 20

    def test_transfer(self):
        assert score_type("TRANSFER") == 20

    def test_payment(self):
        assert score_type("PAYMENT") == 0

    def test_debit(self):
        assert score_type("DEBIT") == 0

    def test_cash_in(self):
        assert score_type("CASH_IN") == 0


# ---------------------------------------------------------------------------
# 4. test_score_balance_drain
# ---------------------------------------------------------------------------

class TestScoreBalanceDrain:
    def test_zero_balance_with_amount(self):
        # old=0, new=0, amount=100 -> 15 * 0.5 = 7.5
        assert score_balance_drain(0, 0, 100) == pytest.approx(7.5)

    def test_zero_balance_zero_amount(self):
        assert score_balance_drain(0, 0, 0) == pytest.approx(0.0)

    def test_full_drain(self):
        # old=1000, new=0, amount=1000 -> drain_ratio=1.0 -> 15
        assert score_balance_drain(1000, 0, 1000) == pytest.approx(15.0)

    def test_drain_ratio_0_9(self):
        # old=1000, new=100, amount=900 -> drain_ratio=0.9 -> 15 * 0.7
        assert score_balance_drain(1000, 100, 900) == pytest.approx(15 * 0.7)

    def test_drain_ratio_0_8(self):
        # old=1000, new=200, amount=800 -> drain_ratio=0.8 -> 15 * 0.7
        assert score_balance_drain(1000, 200, 800) == pytest.approx(15 * 0.7)

    def test_drain_ratio_0_6(self):
        # old=1000, new=400, amount=600 -> drain_ratio=0.6 -> 15 * 0.3
        assert score_balance_drain(1000, 400, 600) == pytest.approx(15 * 0.3)

    def test_drain_ratio_below_threshold(self):
        # old=1000, new=600, amount=400 -> drain_ratio=0.4 -> 0.0
        assert score_balance_drain(1000, 600, 400) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. test_score_balance_mismatch
# ---------------------------------------------------------------------------

class TestScoreBalanceMismatch:
    def test_perfect_match(self):
        # amount=100, old_orig=1000, new_orig=900, old_dest=500, new_dest=600
        assert score_balance_mismatch(100, 1000, 900, 500, 600) == pytest.approx(0.0)

    def test_orig_mismatch_only(self):
        # amount=100, old_orig=1000, new_orig=800, old_dest=500, new_dest=600
        assert score_balance_mismatch(100, 1000, 800, 500, 600) == pytest.approx(15 * 0.5)

    def test_both_mismatch(self):
        # amount=100, old_orig=1000, new_orig=800, old_dest=500, new_dest=700
        assert score_balance_mismatch(100, 1000, 800, 500, 700) == pytest.approx(15.0)

    def test_merchant_dest(self):
        # Merchant dest (both 0): amount=100, old_orig=1000, new_orig=800, old_dest=0, new_dest=0
        assert score_balance_mismatch(100, 1000, 800, 0, 0) == pytest.approx(15 * 0.5)


# ---------------------------------------------------------------------------
# 6. test_compute_dest_frequency_scores
# ---------------------------------------------------------------------------

class TestComputeDestFrequencyScores:
    def test_dest_frequency(self):
        df = pd.DataFrame({"nameDest": ["A", "A", "B"]})
        scores = compute_dest_frequency_scores(df)
        # B appears once -> 10
        assert scores.iloc[2] == pytest.approx(10.0)
        # A appears twice, total=3 -> 10 * max(0, 1 - 2/3)
        expected_a = 10 * max(0, 1 - 2 / 3)
        assert scores.iloc[0] == pytest.approx(expected_a)
        assert scores.iloc[1] == pytest.approx(expected_a)


# ---------------------------------------------------------------------------
# 7. test_compute_rapid_transaction_scores
# ---------------------------------------------------------------------------

class TestComputeRapidTransactionScores:
    def test_rapid_transactions(self):
        df = pd.DataFrame({"nameOrig": ["X", "X", "Y"], "step": [1, 2, 3]})
        scores = compute_rapid_transaction_scores(df)
        # X has count=2, Y has count=1. max_count=2.
        # X: (2-1)/(2-1)*15 = 15
        # Y: (1-1)/(2-1)*15 = 0
        assert scores.iloc[0] == pytest.approx(15.0)
        assert scores.iloc[1] == pytest.approx(15.0)
        assert scores.iloc[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. test_build_explanation
# ---------------------------------------------------------------------------

class TestBuildExplanation:
    def test_non_zero_signals(self):
        result = build_explanation({"a": 5.0, "b": 0.0, "c": 3.0})
        assert result == "a: 5.0; c: 3.0"

    def test_all_zeros(self):
        result = build_explanation({"a": 0.0, "b": 0.0})
        assert result == "No significant risk signals"


# ---------------------------------------------------------------------------
# 9. test_load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_valid_csv(self, tmp_path):
        csv_path = tmp_path / "valid.csv"
        columns = [
            "step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
            "isFraud", "isFlaggedFraud",
        ]
        df = pd.DataFrame(
            [[1, "TRANSFER", 1000, "C1", 5000, 4000, "C2", 100, 1100, 0, 0]],
            columns=columns,
        )
        df.to_csv(csv_path, index=False)
        loaded = load_dataset(str(csv_path))
        assert len(loaded) == 1
        assert set(columns).issubset(set(loaded.columns))

    def test_missing_column(self, tmp_path):
        csv_path = tmp_path / "invalid.csv"
        # Missing "amount" column
        df = pd.DataFrame(
            {
                "step": [1],
                "type": ["TRANSFER"],
                "nameOrig": ["C1"],
                "oldbalanceOrg": [5000],
                "newbalanceOrig": [4000],
                "nameDest": ["C2"],
                "oldbalanceDest": [100],
                "newbalanceDest": [1100],
                "isFraud": [0],
                "isFlaggedFraud": [0],
            }
        )
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError):
            load_dataset(str(csv_path))


# ---------------------------------------------------------------------------
# 10. test_score_transactions
# ---------------------------------------------------------------------------

class TestScoreTransactions:
    def test_output_structure(self):
        df = pd.DataFrame(
            {
                "step": [1, 2],
                "type": ["TRANSFER", "PAYMENT"],
                "amount": [50000, 100],
                "nameOrig": ["C1", "C1"],
                "oldbalanceOrg": [100000, 500],
                "newbalanceOrig": [50000, 400],
                "nameDest": ["C3", "C4"],
                "oldbalanceDest": [0, 200],
                "newbalanceDest": [50000, 300],
                "isFraud": [1, 0],
                "isFlaggedFraud": [0, 0],
            }
        )
        result = score_transactions(df)
        # Verify columns
        assert list(result.columns) == [
            "transaction_id",
            "risk_score",
            "risk_level",
            "explanation",
        ]
        # Verify transaction_id format
        assert result.iloc[0]["transaction_id"] == "TXN-000001"
        # Verify risk_score is between 0 and 100
        assert 0 <= result.iloc[0]["risk_score"] <= 100
        assert 0 <= result.iloc[1]["risk_score"] <= 100


# ---------------------------------------------------------------------------
# 11. test_generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def _make_scored_df(self):
        return pd.DataFrame(
            {
                "transaction_id": ["TXN-000001", "TXN-000002"],
                "risk_score": [85.0, 20.0],
                "risk_level": ["HIGH", "LOW"],
                "explanation": ["high_amount: 25.0", "No significant risk signals"],
            }
        )

    def test_report_content(self):
        scored_df = self._make_scored_df()
        report = generate_report(scored_df)
        assert "FRAUD RISK SCORING REPORT" in report
        assert "Risk Distribution" in report

    def test_report_file_output(self, tmp_path):
        scored_df = self._make_scored_df()
        output_path = tmp_path / "report.txt"
        generate_report(scored_df, output_path=str(output_path))
        assert output_path.exists()
        content = output_path.read_text()
        assert "FRAUD RISK SCORING REPORT" in content
