"""
Parameter Interaction Analyzer
Detects and quantifies non-linear parameter interactions for setup optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score


class InteractionAnalyzer:
    """
    Detect and quantify parameter interactions using polynomial regression.

    Identifies cases where parameter effectiveness depends on other parameter values
    (e.g., "tire pressure more effective when cross weight is higher")
    """

    def __init__(self, max_interaction_order: int = 2, regularization_alpha: float = 1.0):
        """
        Initialize interaction analyzer.

        Args:
            max_interaction_order: Maximum polynomial degree (2 = pairwise interactions)
            regularization_alpha: Ridge regression alpha (higher = more regularization)
        """
        self.max_order = max_interaction_order
        self.alpha = regularization_alpha
        self.significant_interactions = []
        self.model = None
        self.scaler = StandardScaler()

    def find_interactions(
        self,
        df: pd.DataFrame,
        target: str,
        features: List[str],
        significance_threshold: float = 0.05
    ) -> Dict:
        """
        Find significant parameter interactions.

        Args:
            df: Data with setup parameters and lap times
            target: Target variable (usually 'fastest_time')
            features: List of setup parameter names
            significance_threshold: Minimum abs(coefficient) to consider significant

        Returns:
            {
                'interactions': [list of interaction dicts],
                'model_improvement': R² improvement vs linear model,
                'top_interaction': best interaction tuple,
                'linear_r2': baseline linear model R²,
                'poly_r2': polynomial model R²
            }
        """

        if len(df) < 15:
            print(f"   [INTERACTION] Insufficient data for interaction modeling ({len(df)} < 15 samples)")
            return self._empty_result()

        if len(features) < 2:
            print(f"   [INTERACTION] Need at least 2 features for interactions ({len(features)} provided)")
            return self._empty_result()

        # Prepare data
        X = df[features].copy()
        y = df[target].copy()

        # Remove any rows with NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 10:
            print(f"   [INTERACTION] Too few valid samples after NaN removal ({len(X)} < 10)")
            return self._empty_result()

        print(f"   [INTERACTION] Analyzing {len(features)} parameters across {len(X)} sessions")

        # Fit baseline linear model
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        linear_r2 = linear_model.score(X, y)

        print(f"   [BASELINE] Linear model R² = {linear_r2:.3f}")

        # Create polynomial features (interaction terms only)
        poly = PolynomialFeatures(
            degree=self.max_order,
            include_bias=False,
            interaction_only=True  # Only cross-products, no x²
        )
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(features)

        print(f"   [POLY] Created {len(feature_names)} features ({len(feature_names) - len(features)} interactions)")

        # Fit Ridge regression (handles multicollinearity)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_poly, y)
        poly_r2 = self.model.score(X_poly, y)

        print(f"   [POLY] Polynomial model R² = {poly_r2:.3f}")

        # Extract interaction terms
        interactions = []
        for i, (name, coef) in enumerate(zip(feature_names, self.model.coef_)):
            # Interaction terms contain space (e.g., "tire_psi_rr cross_weight")
            if ' ' in name:
                params = tuple(name.split(' '))

                # Only keep significant interactions
                if abs(coef) > significance_threshold:
                    interpretation = self._interpret_interaction(params, coef, features)

                    interactions.append({
                        'params': params,
                        'coefficient': float(coef),
                        'abs_coefficient': float(abs(coef)),
                        'interpretation': interpretation,
                        'synergistic': coef < 0  # Negative coef = synergistic for lap time
                    })

        # Sort by magnitude
        interactions = sorted(interactions, key=lambda x: x['abs_coefficient'], reverse=True)

        self.significant_interactions = interactions

        # Calculate improvement
        improvement = poly_r2 - linear_r2

        if interactions:
            print(f"   [RESULT] Found {len(interactions)} significant interactions")
            print(f"   [RESULT] Model improvement: {improvement:+.3f} R² ({improvement/linear_r2*100:+.1f}%)")

            # Show top 3 interactions
            for i, inter in enumerate(interactions[:3], 1):
                print(f"      {i}. {inter['params'][0]} × {inter['params'][1]}: "
                      f"{inter['coefficient']:+.3f} ({'SYNERGISTIC' if inter['synergistic'] else 'ANTAGONISTIC'})")
        else:
            print(f"   [RESULT] No significant interactions found (all |coef| < {significance_threshold})")

        return {
            'interactions': interactions,
            'model_improvement': float(improvement),
            'improvement_pct': float(improvement / linear_r2 * 100) if linear_r2 > 0 else 0,
            'top_interaction': interactions[0] if interactions else None,
            'linear_r2': float(linear_r2),
            'poly_r2': float(poly_r2),
            'has_significant_interactions': len(interactions) > 0 and improvement > 0.10
        }

    def _interpret_interaction(self, params: Tuple[str, str], coefficient: float, all_features: List[str]) -> str:
        """
        Generate human-readable interpretation of interaction.

        Args:
            params: Tuple of two parameter names
            coefficient: Interaction coefficient
            all_features: All features in model

        Returns:
            Human-readable interpretation string
        """
        param1, param2 = params

        # Clean parameter names for readability
        param1_clean = param1.replace('_', ' ').replace('tire psi', 'tire pressure')
        param2_clean = param2.replace('_', ' ').replace('tire psi', 'tire pressure')

        if coefficient < 0:
            # Negative coefficient = synergistic (both params work together to reduce lap time)
            return (f"{param1_clean} is MORE effective when {param2_clean} is HIGHER "
                   f"(synergistic effect, coefficient: {coefficient:.3f})")
        else:
            # Positive coefficient = antagonistic (params work against each other)
            return (f"{param1_clean} is LESS effective when {param2_clean} is HIGHER "
                   f"(antagonistic effect, coefficient: {coefficient:.3f})")

    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'interactions': [],
            'model_improvement': 0.0,
            'improvement_pct': 0.0,
            'top_interaction': None,
            'linear_r2': 0.0,
            'poly_r2': 0.0,
            'has_significant_interactions': False
        }

    def predict_with_interactions(self, setup: Dict, features: List[str]) -> float:
        """
        Predict lap time using interaction model.

        Args:
            setup: Dictionary of parameter values
            features: Feature names (must match training)

        Returns:
            Predicted lap time
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call find_interactions() first.")

        # Convert dict to dataframe
        X = pd.DataFrame([setup])[features]

        # Create polynomial features
        poly = PolynomialFeatures(degree=self.max_order, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X)

        # Predict
        prediction = self.model.predict(X_poly)[0]

        return float(prediction)

    def get_interaction_strength(self, param1: str, param2: str) -> Optional[Dict]:
        """
        Get interaction strength between two specific parameters.

        Args:
            param1: First parameter name
            param2: Second parameter name

        Returns:
            Interaction dict if exists, None otherwise
        """
        for interaction in self.significant_interactions:
            params_set = set(interaction['params'])
            if params_set == {param1, param2}:
                return interaction

        return None

    def recommend_compound_change(
        self,
        primary_param: str,
        all_impacts: Dict[str, float],
        available_params: List[str]
    ) -> Optional[Dict]:
        """
        Recommend a compound change (two parameters) based on interactions.

        Args:
            primary_param: The main parameter to change (from correlation analysis)
            all_impacts: Dict of parameter impacts on lap time
            available_params: Parameters that can be changed

        Returns:
            {
                'primary_param': str,
                'secondary_param': str,
                'interaction': interaction dict,
                'recommendation': human-readable recommendation
            }
        """
        if not self.significant_interactions:
            return None

        # Find interactions involving primary_param
        relevant_interactions = [
            i for i in self.significant_interactions
            if primary_param in i['params']
        ]

        if not relevant_interactions:
            return None

        # Get the strongest interaction
        best_interaction = relevant_interactions[0]

        # Find the secondary parameter
        secondary_param = [p for p in best_interaction['params'] if p != primary_param][0]

        if secondary_param not in available_params:
            return None

        # Determine direction for both parameters
        primary_impact = all_impacts.get(primary_param, 0)
        secondary_impact = all_impacts.get(secondary_param, 0)

        primary_direction = "REDUCE" if primary_impact > 0 else "INCREASE"
        secondary_direction = "REDUCE" if secondary_impact > 0 else "INCREASE"

        # Generate recommendation
        if best_interaction['synergistic']:
            recommendation = (
                f"COMPOUND RECOMMENDATION:\n"
                f"   Primary: {primary_direction} {primary_param} (impact: {primary_impact:+.3f})\n"
                f"   Synergistic: {secondary_direction} {secondary_param} (impact: {secondary_impact:+.3f})\n"
                f"   \n"
                f"   SYNERGY DETECTED: These parameters work TOGETHER\n"
                f"   {best_interaction['interpretation']}\n"
                f"   \n"
                f"   Expected combined effect: {abs(primary_impact) + abs(secondary_impact) + abs(best_interaction['coefficient']):.3f}s\n"
                f"   (greater than sum of individual effects due to synergy)"
            )
        else:
            recommendation = (
                f"COMPOUND RECOMMENDATION:\n"
                f"   Primary: {primary_direction} {primary_param} (impact: {primary_impact:+.3f})\n"
                f"   Antagonistic: ADJUST {secondary_param} CAREFULLY\n"
                f"   \n"
                f"   CAUTION: These parameters work AGAINST each other\n"
                f"   {best_interaction['interpretation']}\n"
                f"   \n"
                f"   Recommendation: Change {primary_param} FIRST, then fine-tune {secondary_param}"
            )

        return {
            'primary_param': primary_param,
            'secondary_param': secondary_param,
            'interaction': best_interaction,
            'recommendation': recommendation,
            'is_synergistic': best_interaction['synergistic']
        }


def test_interaction_analyzer():
    """Test the interaction analyzer with synthetic data."""
    print("="*70)
    print("  INTERACTION ANALYZER TEST")
    print("="*70)
    print()

    # Create synthetic data with known interaction
    # lap_time = 15.5 + 0.1*tire_psi_rr - 0.08*cross_weight - 0.05*(tire_psi_rr * cross_weight)
    # The interaction term makes tire_psi_rr more effective when cross_weight is higher

    np.random.seed(42)
    n_samples = 30

    tire_psi_rr = np.random.uniform(28, 34, n_samples)
    cross_weight = np.random.uniform(50, 56, n_samples)
    spring_rf = np.random.uniform(400, 600, n_samples)

    # True relationship with interaction
    lap_time = (
        15.5 +
        0.10 * tire_psi_rr +
        -0.08 * cross_weight +
        0.02 * spring_rf +
        -0.05 * (tire_psi_rr * cross_weight / 30)  # Interaction term (normalized)
        + np.random.normal(0, 0.03, n_samples)  # Noise
    )

    df = pd.DataFrame({
        'tire_psi_rr': tire_psi_rr,
        'cross_weight': cross_weight,
        'spring_rf': spring_rf,
        'fastest_time': lap_time
    })

    print(f"Generated {n_samples} synthetic sessions")
    print(f"Lap time range: {lap_time.min():.3f}s - {lap_time.max():.3f}s")
    print()

    # Run interaction analyzer
    analyzer = InteractionAnalyzer()
    result = analyzer.find_interactions(
        df=df,
        target='fastest_time',
        features=['tire_psi_rr', 'cross_weight', 'spring_rf'],
        significance_threshold=0.03
    )

    print()
    print("="*70)
    print("  RESULTS")
    print("="*70)
    print()

    if result['has_significant_interactions']:
        print(f"✓ Significant interactions detected!")
        print(f"   Model improvement: {result['model_improvement']:+.3f} R² ({result['improvement_pct']:+.1f}%)")
        print()
        print("Top interactions:")
        for i, inter in enumerate(result['interactions'][:3], 1):
            print(f"\n{i}. {inter['params'][0]} × {inter['params'][1]}")
            print(f"   Coefficient: {inter['coefficient']:+.3f}")
            print(f"   Type: {'SYNERGISTIC' if inter['synergistic'] else 'ANTAGONISTIC'}")
            print(f"   {inter['interpretation']}")
    else:
        print("✗ No significant interactions detected")
        print(f"   Model improvement: {result['model_improvement']:+.3f} R² (< 10% threshold)")

    print()
    print("="*70)


if __name__ == "__main__":
    test_interaction_analyzer()
