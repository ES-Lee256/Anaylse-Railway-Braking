import unittest

from models import BrakingCase
from runaway import assess_runaway


def make_case(**overrides: float | str) -> BrakingCase:
    values: dict[str, float | str] = {
        "name": "测试工况",
        "mass_kg": 40000.0,
        "initial_speed_kmh": 72.0,
        "braking_ratio": 0.12,
        "response_time_s": 1.0,
        "grade_permille": 0.0,
        "basic_resistance_coefficient": 0.002,
        "rotating_mass_factor": 0.06,
        "adhesion_coefficient": 0.25,
    }
    values.update(overrides)
    return BrakingCase(**values)  # type: ignore[arg-type]


class RunawayAssessmentTests(unittest.TestCase):
    def test_normal_case_can_stop(self) -> None:
        assessment = assess_runaway(make_case())
        self.assertEqual(assessment.classification, "模型内可控停车")
        self.assertFalse(assessment.adhesion_exceeded)
        self.assertGreater(assessment.net_deceleration_m_s2, 0)
        self.assertIsNotNone(assessment.stopping_distance_m)

    def test_adhesion_exceedance_caps_effective_braking_and_reports_risk(self) -> None:
        assessment = assess_runaway(
            make_case(braking_ratio=0.3, adhesion_coefficient=0.1)
        )
        self.assertTrue(assessment.adhesion_exceeded)
        self.assertEqual(assessment.effective_braking_ratio, 0.1)
        self.assertIn("黏着受限", assessment.classification)
        self.assertTrue(any("滑移风险" in finding for finding in assessment.findings))
        self.assertIsNotNone(assessment.stopping_distance_m)

    def test_downhill_with_no_positive_braking_accelerates(self) -> None:
        assessment = assess_runaway(
            make_case(
                braking_ratio=0.01,
                adhesion_coefficient=0.2,
                grade_permille=-20.0,
                basic_resistance_coefficient=0.0,
            )
        )
        self.assertLessEqual(assessment.net_deceleration_m_s2, 0)
        self.assertIn("失控风险", assessment.classification)
        initial_distance, initial_speed = assessment.state_at(1.0)
        later_distance, later_speed = assessment.state_at(5.0)
        self.assertGreater(later_distance, initial_distance)
        self.assertGreater(later_speed, initial_speed)
        self.assertIsNone(assessment.stopping_time_s)

    def test_zero_net_deceleration_is_reported_as_unable_to_stop(self) -> None:
        assessment = assess_runaway(
            make_case(
                braking_ratio=0.01,
                adhesion_coefficient=0.2,
                grade_permille=-10.0,
                basic_resistance_coefficient=0.0,
            )
        )
        self.assertAlmostEqual(assessment.net_deceleration_m_s2, 0.0)
        self.assertIn("失控风险", assessment.classification)
        _, initial_speed = assessment.state_at(1.0)
        _, later_speed = assessment.state_at(5.0)
        self.assertAlmostEqual(initial_speed, later_speed)

    def test_state_stops_at_predicted_distance(self) -> None:
        assessment = assess_runaway(make_case())
        distance, speed = assessment.state_at(assessment.simulation_duration_s)
        self.assertAlmostEqual(distance, assessment.stopping_distance_m)
        self.assertAlmostEqual(speed, 0.0)

    def test_stationary_case_has_no_motion(self) -> None:
        assessment = assess_runaway(make_case(initial_speed_kmh=0.0))
        self.assertEqual(assessment.classification, "静止状态")
        self.assertEqual(assessment.state_at(0.5), (0.0, 0.0))

    def test_invalid_simulation_time_is_rejected(self) -> None:
        assessment = assess_runaway(make_case())
        with self.assertRaisesRegex(ValueError, "仿真时间"):
            assessment.state_at(-1.0)

    def test_long_demo_is_capped_and_marked(self) -> None:
        assessment = assess_runaway(make_case(response_time_s=200.0))
        self.assertEqual(assessment.simulation_duration_s, 120.0)
        self.assertTrue(assessment.simulation_time_limited)
        self.assertAlmostEqual(
            assessment.state_at(assessment.simulation_duration_s)[1],
            assessment.initial_speed_m_s,
        )

    def test_numeric_overflow_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "超出数值范围"):
            assess_runaway(make_case(initial_speed_kmh=1e308))


if __name__ == "__main__":
    unittest.main()
