import math
import unittest

from calculations import GRAVITY_M_S2, calculate_braking
from models import BrakingCase


def make_case(
    *,
    name: str = "测试工况",
    mass_kg: float = 40000.0,
    initial_speed_kmh: float = 72.0,
    braking_ratio: float = 0.12,
    response_time_s: float = 1.0,
    grade_permille: float = 0.0,
    basic_resistance_coefficient: float = 0.002,
    rotating_mass_factor: float = 0.06,
    adhesion_coefficient: float = 0.25,
) -> BrakingCase:
    return BrakingCase(
        name=name,
        mass_kg=mass_kg,
        initial_speed_kmh=initial_speed_kmh,
        braking_ratio=braking_ratio,
        response_time_s=response_time_s,
        grade_permille=grade_permille,
        basic_resistance_coefficient=basic_resistance_coefficient,
        rotating_mass_factor=rotating_mass_factor,
        adhesion_coefficient=adhesion_coefficient,
    )


class BrakingCalculationTests(unittest.TestCase):
    def test_formula_and_unit_conversion(self) -> None:
        result = calculate_braking(make_case())
        initial_speed_m_s = 20.0
        expected_deceleration = (
            GRAVITY_M_S2 * (0.12 + 0.002) / 1.06
        )
        self.assertAlmostEqual(result.deceleration_m_s2, expected_deceleration)
        self.assertAlmostEqual(result.response_distance_m, 20.0)
        self.assertAlmostEqual(
            result.braking_distance_m,
            initial_speed_m_s**2 / (2 * expected_deceleration),
        )
        self.assertAlmostEqual(
            result.stopping_distance_m,
            result.response_distance_m + result.braking_distance_m,
        )

    def test_uphill_increases_and_downhill_reduces_deceleration(self) -> None:
        level = calculate_braking(make_case())
        uphill = calculate_braking(make_case(grade_permille=10.0))
        downhill = calculate_braking(make_case(grade_permille=-10.0))
        self.assertGreater(uphill.deceleration_m_s2, level.deceleration_m_s2)
        self.assertLess(downhill.deceleration_m_s2, level.deceleration_m_s2)

    def test_response_time_adds_linearly_to_distance(self) -> None:
        short = calculate_braking(make_case(response_time_s=1.0))
        long = calculate_braking(make_case(response_time_s=2.0))
        self.assertAlmostEqual(
            long.stopping_distance_m - short.stopping_distance_m, 20.0
        )

    def test_zero_speed_has_zero_stopping_time_and_distance(self) -> None:
        result = calculate_braking(make_case(initial_speed_kmh=0.0))
        self.assertEqual(result.total_time_s, 0.0)
        self.assertEqual(result.stopping_distance_m, 0.0)
        self.assertEqual(result.curve_points[-1].speed_m_s, 0.0)

    def test_adhesion_limit_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "黏着系数"):
            calculate_braking(make_case(adhesion_coefficient=0.1))

    def test_nonpositive_deceleration_is_rejected(self) -> None:
        case = make_case(
            braking_ratio=0.01,
            adhesion_coefficient=0.2,
            grade_permille=-20.0,
            basic_resistance_coefficient=0.0,
        )
        with self.assertRaisesRegex(ValueError, "减速度不大于 0"):
            calculate_braking(case)

    def test_curve_ends_at_calculated_stop(self) -> None:
        result = calculate_braking(make_case())
        final_point = result.curve_points[-1]
        self.assertAlmostEqual(final_point.time_s, result.total_time_s)
        self.assertAlmostEqual(final_point.distance_m, result.stopping_distance_m)
        self.assertAlmostEqual(final_point.speed_m_s, 0.0)
        self.assertTrue(
            all(
                left.distance_m <= right.distance_m
                for left, right in zip(
                    result.curve_points, result.curve_points[1:]
                )
            )
        )

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "车辆质量"):
            make_case(mass_kg=0.0)
        with self.assertRaisesRegex(ValueError, "制动率"):
            make_case(braking_ratio=-0.1)
        with self.assertRaisesRegex(ValueError, "有限数值"):
            make_case(initial_speed_kmh=math.inf)

    def test_numeric_overflow_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "超出数值范围"):
            calculate_braking(make_case(mass_kg=1e308))


if __name__ == "__main__":
    unittest.main()
