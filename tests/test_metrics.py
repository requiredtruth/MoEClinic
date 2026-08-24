import math
import unittest

from moeclinic.metrics import analyze
from moeclinic.trace import TraceBatch


class MetricsTests(unittest.TestCase):
    def test_balanced_router_has_expected_auxiliary_and_z_losses(self) -> None:
        trace = TraceBatch(
            step=1,
            token_count=100,
            assignment_counts=(25, 25, 25, 25),
            probability_sums=(25.0, 25.0, 25.0, 25.0),
            logsumexp_sq_sum=100 * math.log(4) ** 2,
            logit_token_count=100,
        )
        report = analyze([trace])
        self.assertAlmostEqual(report.load_balance_loss, 0.01)
        self.assertAlmostEqual(report.z_loss or 0.0, 1e-4 * math.log(4) ** 2)
        self.assertAlmostEqual(report.normalized_entropy, 1.0)
        self.assertEqual(report.severity, "healthy")

    def test_collapsed_router_is_detected(self) -> None:
        report = analyze(
            [
                TraceBatch(
                    step=1,
                    token_count=100,
                    assignment_counts=(90, 10, 0, 0),
                    probability_sums=(70.0, 20.0, 5.0, 5.0),
                )
            ]
        )
        self.assertEqual(report.severity, "collapse")
        self.assertEqual(report.dead_experts, (2, 3))
        self.assertEqual(report.max_expert, 0)
        self.assertAlmostEqual(report.max_share, 0.9)
        self.assertGreater(report.load_balance_loss, 0.01)

    def test_capacity_overflow_and_drops_are_reported(self) -> None:
        report = analyze(
            [
                TraceBatch(
                    step=1,
                    token_count=10,
                    assignment_counts=(5, 5),
                    probability_sums=(5.0, 5.0),
                    capacity=4,
                    dropped_assignments=2,
                )
            ]
        )
        self.assertEqual(report.capacity_overflow, 2)
        self.assertEqual(report.dropped_assignments, 2)
        self.assertEqual(report.severity, "warning")


if __name__ == "__main__":
    unittest.main()
