"""
Recommendation Deduplication System

Prevents agents from suggesting the same setup changes multiple times.
"""

from typing import Dict, List, Any, Optional, Tuple
import math


class RecommendationDeduplicator:
    """Detects and filters duplicate recommendations"""

    def __init__(self, tolerance: float = 0.1):
        """
        Initialize deduplicator.

        Args:
            tolerance: Fractional tolerance for considering recommendations the same
                      (e.g., 0.1 = within 10% is considered duplicate)
        """
        self.tolerance = tolerance

    def is_duplicate(
        self,
        proposed: Dict[str, Any],
        previous: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if proposed recommendation duplicates a previous one.

        Args:
            proposed: New recommendation dict with parameter, direction, magnitude
            previous: List of previous recommendations

        Returns:
            Tuple of (is_duplicate: bool, matching_rec: Optional[Dict])
        """
        if not previous:
            return False, None

        prop_param = proposed.get('parameter')
        prop_direction = proposed.get('direction', '').lower()
        prop_magnitude = proposed.get('magnitude', 0)

        for prev_rec in previous:
            prev_param = prev_rec.get('parameter')
            prev_direction = prev_rec.get('direction', '').lower()
            prev_magnitude = prev_rec.get('magnitude', 0)

            # Same parameter?
            if prop_param != prev_param:
                continue

            # Same direction?
            if prop_direction != prev_direction:
                continue

            # Similar magnitude? (within tolerance)
            if self._magnitudes_similar(prop_magnitude, prev_magnitude):
                return True, prev_rec

        return False, None

    def filter_duplicates(
        self,
        proposed_recommendations: List[Dict[str, Any]],
        previous_recommendations: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter out duplicates from proposed recommendations.

        Args:
            proposed_recommendations: New recommendations to check
            previous_recommendations: Historical recommendations

        Returns:
            Tuple of (unique_recs, duplicate_recs)
        """
        unique = []
        duplicates = []

        # Also check against other proposed recommendations
        all_previous = previous_recommendations.copy()

        for proposed in proposed_recommendations:
            is_dup, matching = self.is_duplicate(proposed, all_previous)

            if is_dup:
                proposed['duplicate_of'] = matching
                proposed['filtered_reason'] = 'duplicate'
                duplicates.append(proposed)
            else:
                unique.append(proposed)
                # Add to previous to prevent duplicates within this batch
                all_previous.append(proposed)

        return unique, duplicates

    def _magnitudes_similar(self, mag1: float, mag2: float) -> bool:
        """Check if two magnitudes are similar within tolerance"""
        if mag1 == 0 and mag2 == 0:
            return True

        if mag1 == 0 or mag2 == 0:
            # One is zero, other is not - check absolute difference
            return abs(mag1 - mag2) < 0.5  # Small absolute tolerance

        # Check relative difference
        relative_diff = abs(mag1 - mag2) / max(abs(mag1), abs(mag2))
        return relative_diff <= self.tolerance

    def find_similar_recommendations(
        self,
        proposed: Dict[str, Any],
        previous: List[Dict[str, Any]],
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find recommendations that are similar but not exact duplicates.

        Useful for suggesting alternatives.

        Args:
            proposed: Proposed recommendation
            previous: Previous recommendations
            similarity_threshold: How similar (0-1) to be considered similar

        Returns:
            List of similar recommendations
        """
        similar = []
        prop_param = proposed.get('parameter', '')

        for prev in previous:
            prev_param = prev.get('parameter', '')

            # Same parameter type? (e.g., both tire pressures)
            if self._parameters_related(prop_param, prev_param):
                similarity = self._calculate_similarity(proposed, prev)
                if similarity >= similarity_threshold and similarity < 1.0:
                    prev_copy = prev.copy()
                    prev_copy['similarity_score'] = similarity
                    similar.append(prev_copy)

        return sorted(similar, key=lambda x: x['similarity_score'], reverse=True)

    def _parameters_related(self, param1: str, param2: str) -> bool:
        """Check if two parameters are related (same subsystem)"""
        # Group parameters by subsystem
        subsystems = {
            'tire_pressure': ['tire_psi_lf', 'tire_psi_rf', 'tire_psi_lr', 'tire_psi_rr'],
            'springs': ['spring_lf', 'spring_rf', 'spring_lr', 'spring_rr'],
            'chassis': ['cross_weight', 'track_bar_height_left'],
        }

        for subsystem_params in subsystems.values():
            if param1 in subsystem_params and param2 in subsystem_params:
                return True

        return param1 == param2

    def _calculate_similarity(self, rec1: Dict, rec2: Dict) -> float:
        """Calculate similarity score between two recommendations (0-1)"""
        score = 0.0

        # Same parameter = 0.5
        if rec1.get('parameter') == rec2.get('parameter'):
            score += 0.5

        # Same direction = 0.3
        if rec1.get('direction', '').lower() == rec2.get('direction', '').lower():
            score += 0.3

        # Similar magnitude = 0.2
        mag1 = rec1.get('magnitude', 0)
        mag2 = rec2.get('magnitude', 0)
        if mag1 and mag2:
            mag_similarity = 1.0 - min(abs(mag1 - mag2) / max(abs(mag1), abs(mag2)), 1.0)
            score += 0.2 * mag_similarity

        return score

    def suggest_alternatives(
        self,
        rejected_recommendation: Dict[str, Any],
        statistical_analysis: Optional[Dict[str, Any]] = None,
        previous_recommendations: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest alternative recommendations when one is rejected as duplicate.

        Args:
            rejected_recommendation: The duplicate recommendation
            statistical_analysis: Statistical results to find alternatives
            previous_recommendations: What's already been tried

        Returns:
            List of alternative recommendation suggestions
        """
        alternatives = []
        rejected_param = rejected_recommendation.get('parameter')

        if not statistical_analysis:
            return alternatives

        # Find next best parameters that haven't been tried
        all_impacts = statistical_analysis.get('correlations') or statistical_analysis.get('coefficients', {})
        sorted_params = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

        already_tried = {rejected_param}
        if previous_recommendations:
            already_tried.update(rec.get('parameter') for rec in previous_recommendations)

        for param, impact in sorted_params:
            if param not in already_tried:
                # Create alternative recommendation
                direction = "decrease" if impact > 0 else "increase"

                # Estimate magnitude based on parameter type
                magnitude, unit = self._estimate_magnitude(param)

                alternatives.append({
                    'parameter': param,
                    'direction': direction,
                    'magnitude': magnitude,
                    'magnitude_unit': unit,
                    'impact': impact,
                    'rationale': f"Next most impactful parameter (impact: {impact:.3f})",
                    'confidence': 0.7,
                    'is_alternative': True
                })

                if len(alternatives) >= 3:
                    break

        return alternatives

    def _estimate_magnitude(self, parameter: str) -> Tuple[float, str]:
        """Estimate appropriate magnitude for a parameter"""
        if 'psi' in parameter.lower() or 'tire' in parameter.lower():
            return 1.5, "PSI"
        elif 'spring' in parameter.lower():
            return 25.0, "lb/in"
        elif 'cross_weight' in parameter.lower() or 'weight' in parameter.lower():
            return 0.5, "%"
        elif 'track_bar' in parameter.lower() or 'height' in parameter.lower():
            return 0.25, "inches"
        else:
            return 1.0, "units"


# Singleton instance
_deduplicator = RecommendationDeduplicator()


def check_duplicate(
    proposed: Dict[str, Any],
    previous: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convenience function to check for duplicates.

    Returns dict with:
        - is_duplicate: bool
        - matching_recommendation: Optional[Dict]
        - similar_recommendations: List[Dict]
    """
    is_dup, matching = _deduplicator.is_duplicate(proposed, previous)
    similar = _deduplicator.find_similar_recommendations(proposed, previous) if not is_dup else []

    return {
        "is_duplicate": is_dup,
        "matching_recommendation": matching,
        "similar_recommendations": similar
    }


def filter_unique(
    proposed: List[Dict[str, Any]],
    previous: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convenience function to filter duplicates.

    Returns dict with:
        - unique: List[Dict]
        - duplicates: List[Dict]
        - num_filtered: int
    """
    unique, dups = _deduplicator.filter_duplicates(proposed, previous)

    return {
        "unique": unique,
        "duplicates": dups,
        "num_filtered": len(dups)
    }
