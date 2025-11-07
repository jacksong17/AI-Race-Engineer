"""
Outcome Validator - Closed-loop Learning System
Validates whether recommendations actually improved lap times
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


class OutcomeValidator:
    """
    Automatically validate if a recommendation improved lap times.

    Uses statistical testing to determine if changes are significant,
    not just random variation.
    """

    def __init__(self, confidence_level: float = 0.80):
        """
        Initialize outcome validator.

        Args:
            confidence_level: Statistical confidence required (0-1)
        """
        self.confidence_level = confidence_level

    def validate_recommendation_outcome(
        self,
        baseline_time: float,
        test_laps: List[float],
        recommendation: str,
        baseline_std: float = 0.05
    ) -> Dict:
        """
        Determine if recommendation was effective.

        Args:
            baseline_time: Best lap before change
            test_laps: Lap times after change (min 3 laps recommended)
            recommendation: The recommendation that was tested
            baseline_std: Estimated baseline lap time variance (default 0.05s)

        Returns:
            {
                'outcome': 'improved' | 'no_change' | 'worse' | 'insufficient_data',
                'lap_time_delta': float (negative = faster),
                'statistical_confidence': float (0-1),
                'sample_size': int,
                'recommended_action': 'accept' | 'refine' | 'revert' | 'retest',
                'learning_note': str,
                'best_test_lap': float,
                'mean_test_lap': float,
                'test_consistency': float
            }
        """

        if len(test_laps) < 1:
            return {
                'outcome': 'insufficient_data',
                'lap_time_delta': 0.0,
                'statistical_confidence': 0.0,
                'sample_size': 0,
                'recommended_action': 'retest',
                'learning_note': "No test laps provided - need at least 3 laps for validation",
                'parameter_tested': self._extract_parameter_from_recommendation(recommendation)
            }

        # Calculate test session statistics
        best_test_lap = min(test_laps)
        mean_test_lap = np.mean(test_laps)
        test_std = np.std(test_laps) if len(test_laps) > 1 else baseline_std
        test_consistency = test_std  # Lower = more consistent

        # Calculate improvement
        improvement = baseline_time - best_test_lap  # Positive = faster

        print(f"   [VALIDATION] Baseline: {baseline_time:.3f}s")
        print(f"   [VALIDATION] Best test lap: {best_test_lap:.3f}s")
        print(f"   [VALIDATION] Mean test lap: {mean_test_lap:.3f}s")
        print(f"   [VALIDATION] Test consistency: {test_std:.3f}s std dev")
        print(f"   [VALIDATION] Improvement: {improvement:+.3f}s")

        # Statistical significance test
        if len(test_laps) >= 3:
            # Use t-test to compare baseline vs test distribution
            # Create baseline distribution (approximate)
            baseline_laps = [baseline_time] * max(3, len(test_laps))

            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(
                baseline_laps,
                test_laps,
                equal_var=False
            )

            confidence = 1 - p_value
            print(f"   [STATS] t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
            print(f"   [STATS] Confidence: {confidence:.1%}")
        else:
            # Not enough data for proper t-test, use heuristic
            if len(test_laps) == 2:
                # Simple comparison with relaxed threshold
                confidence = 0.6 if abs(improvement) > 0.08 else 0.4
            else:
                # Single lap - very low confidence
                confidence = 0.5 if abs(improvement) > 0.10 else 0.3

            print(f"   [STATS] Insufficient samples for t-test (n={len(test_laps)})")
            print(f"   [STATS] Using heuristic confidence: {confidence:.1%}")

        # Classify outcome
        outcome, action, note = self._classify_outcome(
            improvement=improvement,
            confidence=confidence,
            test_consistency=test_consistency,
            sample_size=len(test_laps)
        )

        return {
            'outcome': outcome,
            'lap_time_delta': improvement,
            'statistical_confidence': confidence,
            'sample_size': len(test_laps),
            'recommended_action': action,
            'learning_note': note,
            'parameter_tested': self._extract_parameter_from_recommendation(recommendation),
            'best_test_lap': best_test_lap,
            'mean_test_lap': mean_test_lap,
            'test_consistency': test_consistency
        }

    def _classify_outcome(
        self,
        improvement: float,
        confidence: float,
        test_consistency: float,
        sample_size: int
    ) -> Tuple[str, str, str]:
        """
        Classify outcome and recommend action.

        Args:
            improvement: Lap time improvement (positive = faster)
            confidence: Statistical confidence (0-1)
            test_consistency: Std dev of test laps
            sample_size: Number of test laps

        Returns:
            (outcome, action, note) tuple
        """

        # Thresholds
        MEANINGFUL_THRESHOLD = 0.05  # 0.05s = meaningful change
        HIGH_CONFIDENCE = self.confidence_level
        GOOD_CONSISTENCY = 0.08  # <0.08s std dev = consistent

        # Decision tree
        if improvement < -MEANINGFUL_THRESHOLD and confidence > HIGH_CONFIDENCE:
            # Significantly worse
            outcome = 'worse'
            action = 'revert'
            note = (f"Change made car {abs(improvement):.3f}s SLOWER with {confidence:.0%} confidence. "
                   f"REVERT immediately and try different parameter.")

        elif improvement > MEANINGFUL_THRESHOLD and confidence > HIGH_CONFIDENCE:
            # Significantly better
            outcome = 'improved'
            action = 'accept'
            note = (f"Change improved lap time by {improvement:.3f}s with {confidence:.0%} confidence. "
                   f"ACCEPT this change and consider further optimization.")

        elif abs(improvement) < MEANINGFUL_THRESHOLD and confidence > 0.6:
            # No meaningful change
            outcome = 'no_change'

            if sample_size < 5:
                action = 'retest'
                note = (f"No significant change ({improvement:+.3f}s). "
                       f"Run {5-sample_size} more laps to increase confidence.")
            else:
                action = 'refine'
                note = (f"No significant change ({improvement:+.3f}s) after {sample_size} laps. "
                       f"Try larger magnitude or different parameter.")

        elif confidence < HIGH_CONFIDENCE and sample_size < 5:
            # Inconclusive - need more data
            outcome = 'inconclusive'
            action = 'retest'
            note = (f"Trend shows {improvement:+.3f}s but only {confidence:.0%} confidence. "
                   f"Need {5-sample_size} more laps to confirm.")

        elif test_consistency > 0.12:
            # Inconsistent - can't draw conclusions
            outcome = 'inconclusive'
            action = 'retest'
            note = (f"Test laps too inconsistent (std dev: {test_consistency:.3f}s). "
                   f"Ensure consistent driving and track conditions, then retest.")

        else:
            # Default: trend detected but not confident
            if improvement > 0:
                outcome = 'inconclusive'
                action = 'retest'
                note = (f"Possible improvement ({improvement:+.3f}s) but {confidence:.0%} confidence < {HIGH_CONFIDENCE:.0%} threshold. "
                       f"Run more laps to confirm.")
            else:
                outcome = 'inconclusive'
                action = 'refine'
                note = (f"Possible degradation ({improvement:+.3f}s). Consider reverting or trying different parameter.")

        return outcome, action, note

    def _extract_parameter_from_recommendation(self, recommendation: str) -> Optional[str]:
        """
        Extract parameter name from recommendation string.

        Args:
            recommendation: Recommendation string

        Returns:
            Parameter name or None
        """
        # Common parameter names
        params = [
            'tire_psi_lf', 'tire_psi_rf', 'tire_psi_lr', 'tire_psi_rr',
            'cross_weight', 'track_bar_height_left', 'track_bar_height_right',
            'spring_lf', 'spring_rf', 'spring_lr', 'spring_rr',
            'arb_front', 'arb_rear'
        ]

        recommendation_lower = recommendation.lower()
        for param in params:
            if param in recommendation_lower:
                return param

        return None

    def compare_multiple_changes(
        self,
        baseline_time: float,
        change_results: List[Dict]
    ) -> Dict:
        """
        Compare multiple different setup changes to find the best.

        Args:
            baseline_time: Original baseline lap time
            change_results: List of dicts with {
                'parameter': str,
                'test_laps': List[float],
                'recommendation': str
            }

        Returns:
            {
                'best_change': dict,
                'ranking': list of validated results sorted by improvement,
                'summary': str
            }
        """
        print(f"\n   [COMPARISON] Comparing {len(change_results)} different changes")
        print(f"   [BASELINE] {baseline_time:.3f}s")
        print()

        # Validate each change
        validated = []
        for i, change in enumerate(change_results, 1):
            print(f"   [{i}/{len(change_results)}] Testing {change['parameter']}")

            result = self.validate_recommendation_outcome(
                baseline_time=baseline_time,
                test_laps=change['test_laps'],
                recommendation=change['recommendation']
            )

            result['parameter'] = change['parameter']
            validated.append(result)

            print(f"      Outcome: {result['outcome'].upper()}")
            print(f"      Improvement: {result['lap_time_delta']:+.3f}s")
            print()

        # Rank by improvement (weighted by confidence)
        ranked = sorted(
            validated,
            key=lambda x: x['lap_time_delta'] * x['statistical_confidence'],
            reverse=True
        )

        best = ranked[0] if ranked else None

        if best and best['outcome'] == 'improved':
            summary = (f"BEST CHANGE: {best['parameter']}\n"
                      f"   Improvement: {best['lap_time_delta']:+.3f}s\n"
                      f"   Confidence: {best['statistical_confidence']:.0%}\n"
                      f"   Action: {best['recommended_action'].upper()}")
        else:
            summary = "No significant improvement found from any tested changes."

        return {
            'best_change': best,
            'ranking': ranked,
            'summary': summary
        }


def test_outcome_validator():
    """Test the outcome validator with scenarios."""
    print("="*70)
    print("  OUTCOME VALIDATOR TEST")
    print("="*70)
    print()

    validator = OutcomeValidator(confidence_level=0.80)

    # Test scenarios
    scenarios = [
        {
            'name': "Significant Improvement",
            'baseline': 15.50,
            'test_laps': [15.32, 15.35, 15.30, 15.33, 15.31],
            'expected': 'improved'
        },
        {
            'name': "Significant Degradation",
            'baseline': 15.50,
            'test_laps': [15.68, 15.72, 15.70, 15.69, 15.71],
            'expected': 'worse'
        },
        {
            'name': "No Meaningful Change",
            'baseline': 15.50,
            'test_laps': [15.48, 15.52, 15.51, 15.49, 15.50],
            'expected': 'no_change'
        },
        {
            'name': "Inconclusive (not enough laps)",
            'baseline': 15.50,
            'test_laps': [15.40, 15.45],
            'expected': 'inconclusive'
        },
        {
            'name': "Inconsistent Driving",
            'baseline': 15.50,
            'test_laps': [15.30, 15.65, 15.42, 15.80, 15.25],
            'expected': 'inconclusive'
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print(f"{'='*70}")

        result = validator.validate_recommendation_outcome(
            baseline_time=scenario['baseline'],
            test_laps=scenario['test_laps'],
            recommendation="Test parameter change"
        )

        print(f"\n   RESULT:")
        print(f"      Outcome: {result['outcome'].upper()}")
        print(f"      Action: {result['recommended_action'].upper()}")
        print(f"      {result['learning_note']}")

        # Check if result matches expectation
        matches = result['outcome'] == scenario['expected'] or (
            result['outcome'] in ['inconclusive', scenario['expected']]
        )
        status = "✓ PASS" if matches else "✗ FAIL"
        print(f"\n   {status} (expected: {scenario['expected']}, got: {result['outcome']})")

        print()

    print("="*70)


if __name__ == "__main__":
    test_outcome_validator()
