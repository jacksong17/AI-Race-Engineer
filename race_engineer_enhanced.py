"""
Enhanced Race Engineer with Real Agent Decision-Making
Demonstrates: Model selection, feature engineering, data quality assessment
"""

import pandas as pd
from typing import TypedDict, Optional, Dict, List
from langgraph.graph import StateGraph, END
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np


class RaceEngineerState(TypedDict, total=False):
    raw_setup_data: pd.DataFrame
    data_quality_report: Optional[Dict]
    selected_features: Optional[List[str]]
    analysis_strategy: Optional[str]
    analysis: Optional[Dict]
    recommendation: Optional[str]
    error: Optional[str]


# ========== AGENT 1: DATA QUALITY ASSESSOR ==========
def data_quality_agent(state: RaceEngineerState) -> Dict:
    """
    Agent 1: Assesses data quality and makes filtering decisions
    DECISION POINTS:
    - Remove outliers or keep them?
    - Sufficient sample size?
    - Data variance adequate for modeling?
    """
    print("\n[AGENT 1] Data Quality Assessor: Evaluating dataset...")

    df = state.get('raw_setup_data')
    if df is None or df.empty:
        return {'error': 'No data provided'}

    # Decision 1: Outlier detection
    lap_times = df['fastest_time']
    q1, q3 = lap_times.quantile(0.25), lap_times.quantile(0.75)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    outliers = df[lap_times > outlier_threshold]

    print(f"   📊 Dataset size: {len(df)} sessions")
    print(f"   📈 Lap time range: {lap_times.min():.3f}s - {lap_times.max():.3f}s")
    print(f"   🎯 Lap time variance: {lap_times.std():.3f}s")

    # Agent Decision: Should we filter outliers?
    if len(outliers) > 0:
        print(f"   ⚠️  Detected {len(outliers)} outlier sessions (>{outlier_threshold:.3f}s)")
        if len(outliers) < len(df) * 0.2:  # Less than 20% outliers
            df_clean = df[lap_times <= outlier_threshold]
            print(f"    DECISION: Removing outliers (keeping {len(df_clean)} sessions)")
            decision = "removed_outliers"
        else:
            print(f"    DECISION: Keeping all data (too many outliers to remove)")
            df_clean = df
            decision = "kept_outliers"
    else:
        print(f"    DECISION: No outliers detected, using all data")
        df_clean = df
        decision = "no_outliers"

    # Decision 2: Sample size assessment
    if len(df_clean) < 5:
        return {'error': f'Insufficient data: Only {len(df_clean)} valid sessions'}
    elif len(df_clean) < 10:
        analysis_recommendation = "correlation"  # Use simpler method
        print(f"   📉 Sample size small ({len(df_clean)}), recommend correlation analysis")
    else:
        analysis_recommendation = "regression"  # Can use complex model
        print(f"   📈 Sample size adequate ({len(df_clean)}), can use regression")

    # Decision 3: Variance check
    if lap_times.std() < 0.1:
        print(f"   ⚠️  Low variance detected - setup changes may have minimal impact")

    quality_report = {
        'original_size': len(df),
        'final_size': len(df_clean),
        'outliers_removed': len(outliers),
        'filtering_decision': decision,
        'variance': float(lap_times.std()),
        'recommended_strategy': analysis_recommendation
    }

    return {
        'raw_setup_data': df_clean,
        'data_quality_report': quality_report
    }


# ========== AGENT 2: FEATURE SELECTOR ==========
def feature_selection_agent(state: RaceEngineerState) -> Dict:
    """
    Agent 2: Decides which setup parameters to analyze
    DECISION POINTS:
    - Which features have enough variance?
    - Which features are independent?
    - Which features to include in model?
    """
    print("\n[AGENT 2] Feature Selector: Analyzing setup parameters...")

    df = state.get('raw_setup_data')
    if df is None or df.empty:
        return {'error': 'No data from quality assessment'}

    # All possible features
    potential_features = [
        'tire_psi_lf', 'tire_psi_rf', 'tire_psi_lr', 'tire_psi_rr',
        'cross_weight', 'track_bar_height_left',
        'spring_lf', 'spring_rf', 'spring_lr', 'spring_rr'
    ]

    # Agent Decision: Which features to use?
    selected = []
    rejected = []

    for feature in potential_features:
        if feature not in df.columns:
            rejected.append((feature, 'missing'))
            continue

        # Decision criteria 1: Has variance (was actually changed)
        variance = df[feature].std()
        if variance < 0.01:  # Essentially constant
            rejected.append((feature, f'no_variance ({variance:.4f})'))
            continue

        # Decision criteria 2: Has values (not all NaN)
        if df[feature].isna().sum() > len(df) * 0.5:
            rejected.append((feature, 'too_many_missing'))
            continue

        # Feature passes all checks
        selected.append(feature)

    print(f"   🎯 Selected {len(selected)} features for analysis:")
    for feat in selected:
        var = df[feat].std()
        print(f"       {feat:25s} (variance: {var:.3f})")

    if rejected:
        print(f"   [X] Rejected {len(rejected)} features:")
        for feat, reason in rejected[:3]:  # Show first 3
            print(f"      [X] {feat:25s} ({reason})")

    if len(selected) < 2:
        return {'error': f'Not enough variable features ({len(selected)} found)'}

    # Additional agent decision: Warn about collinearity
    if 'tire_psi_rf' in selected and 'tire_psi_rr' in selected:
        corr = df[['tire_psi_rf', 'tire_psi_rr']].corr().iloc[0, 1]
        if abs(corr) > 0.9:
            print(f"   ⚠️  High correlation between RF and RR pressures ({corr:.2f})")
            print(f"   ℹ️  Consider testing these independently")

    return {
        'selected_features': selected,
        'feature_count': len(selected)
    }


# ========== AGENT 3: MODEL STRATEGIST ==========
def model_selection_agent(state: RaceEngineerState) -> Dict:
    """
    Agent 3: Chooses analytical strategy and runs analysis
    DECISION POINTS:
    - Regression, correlation, or decision tree?
    - Based on: sample size, feature count, data quality
    """
    print("\n[AGENT 3] Model Strategist: Selecting analysis approach...")

    df = state.get('raw_setup_data')
    features = state.get('selected_features', [])
    quality_report = state.get('data_quality_report', {})

    if df is None or not features:
        return {'error': 'Missing data or features'}

    # Agent Decision: Choose strategy
    sample_size = len(df)
    feature_count = len(features)
    variance = quality_report.get('variance', 0)

    print(f"   📊 Analysis context:")
    print(f"      • Sample size: {sample_size}")
    print(f"      • Features: {feature_count}")
    print(f"      • Variance: {variance:.3f}s")

    # Decision logic
    if sample_size < 10:
        strategy = "correlation"
        print(f"    DECISION: Using CORRELATION (small sample)")
    elif feature_count > sample_size / 2:
        strategy = "correlation"
        print(f"    DECISION: Using CORRELATION (many features vs samples)")
    elif variance < 0.15:
        strategy = "correlation"
        print(f"    DECISION: Using CORRELATION (low variance)")
    else:
        strategy = "regression"
        print(f"    DECISION: Using REGRESSION (adequate data quality)")

    # Execute chosen strategy
    try:
        if strategy == "correlation":
            results = _run_correlation_analysis(df, features)
        else:
            results = _run_regression_analysis(df, features)

        results['strategy_used'] = strategy
        return {'analysis': results, 'analysis_strategy': strategy}

    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}'}


def _run_correlation_analysis(df: pd.DataFrame, features: List[str]) -> Dict:
    """Simple correlation analysis"""
    print(f"   🔬 Running correlation analysis...")

    target = 'fastest_time'
    correlations = {}

    for feature in features:
        corr = df[[feature, target]].corr().iloc[0, 1]
        correlations[feature] = float(corr)

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"   📈 Correlation results:")
    for feature, corr in sorted_corr[:5]:
        direction = "faster" if corr < 0 else "slower"
        print(f"      • {feature:25s}: {corr:+.3f} ({direction} when increased)")

    return {
        'method': 'correlation',
        'impacts': correlations,
        'sorted_impacts': sorted_corr
    }


def _run_regression_analysis(df: pd.DataFrame, features: List[str]) -> Dict:
    """Full regression analysis"""
    print(f"   🔬 Running regression analysis...")

    target = 'fastest_time'
    X = df[features].fillna(df[features].mean())
    y = df[target]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Get coefficients
    impacts = {feat: float(coef) for feat, coef in zip(features, model.coef_)}
    sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"   📈 Regression results (R² = {model.score(X_scaled, y):.3f}):")
    for feature, impact in sorted_impacts[:5]:
        direction = "REDUCE" if impact > 0 else "INCREASE"
        print(f"      • {feature:25s}: {impact:+.3f} ({direction} to improve)")

    return {
        'method': 'regression',
        'impacts': impacts,
        'sorted_impacts': sorted_impacts,
        'r_squared': float(model.score(X_scaled, y))
    }


# ========== AGENT 4: RECOMMENDATION SYNTHESIZER ==========
def recommendation_agent(state: RaceEngineerState) -> Dict:
    """
    Agent 4: Creates actionable recommendations
    DECISION POINTS:
    - Single parameter or multi-parameter change?
    - Magnitude of recommended change?
    - Priority ordering?
    """
    print("\n[AGENT 4] Recommendation Synthesizer: Generating advice...")

    analysis = state.get('analysis', {})
    strategy = state.get('analysis_strategy', 'unknown')

    if not analysis:
        return {'error': 'No analysis results'}

    impacts = analysis.get('impacts', {})
    sorted_impacts = analysis.get('sorted_impacts', [])

    if not sorted_impacts:
        return {'recommendation': 'No significant correlations found'}

    # Agent Decision: Single or multi-parameter recommendation?
    top_param, top_impact = sorted_impacts[0]

    print(f"   🎯 Analysis method: {strategy.upper()}")
    print(f"   📊 Top finding: {top_param} ({top_impact:+.3f})")

    # Decision logic for recommendation
    if abs(top_impact) > 0.1:
        # Strong single-parameter effect
        if strategy == "correlation":
            if top_impact < -0.3:
                rec = f"STRONG SIGNAL: Increase {top_param} significantly (correlation: {top_impact:.3f})"
            elif top_impact > 0.3:
                rec = f"STRONG SIGNAL: Reduce {top_param} significantly (correlation: {top_impact:.3f})"
            else:
                direction = "increase" if top_impact < 0 else "reduce"
                rec = f"Moderate signal: {direction} {top_param} (correlation: {top_impact:.3f})"
        else:  # regression
            direction = "REDUCE" if top_impact > 0 else "INCREASE"
            rec = f"Primary recommendation: {direction} {top_param} (impact: {top_impact:+.3f}s per unit)"

    else:
        # Weak effects - recommend multi-parameter approach
        top_3 = sorted_impacts[:3]
        params = [p for p, _ in top_3]
        rec = f"No dominant parameter. Test interaction between: {', '.join(params)}"

    print(f"    DECISION: {rec}")

    # Additional insights
    if len(sorted_impacts) > 3:
        print(f"   ℹ️  Secondary factors to monitor:")
        for param, impact in sorted_impacts[1:4]:
            print(f"      • {param}: {impact:+.3f}")

    return {'recommendation': rec}


# ========== BUILD GRAPH ==========
def create_enhanced_workflow():
    """Build modular agent workflow with decision points"""

    workflow = StateGraph(RaceEngineerState)

    # Add agent nodes
    workflow.add_node("data_quality", data_quality_agent)
    workflow.add_node("feature_selection", feature_selection_agent)
    workflow.add_node("model_selection", model_selection_agent)
    workflow.add_node("recommendation", recommendation_agent)
    workflow.add_node("error_handler", lambda s: s)

    # Conditional routing based on agent decisions
    def check_data_quality(state):
        return "error_handler" if state.get('error') else "feature_selection"

    def check_features(state):
        return "error_handler" if state.get('error') else "model_selection"

    def check_analysis(state):
        return "error_handler" if state.get('error') else "recommendation"

    # Build graph
    workflow.set_entry_point("data_quality")
    workflow.add_conditional_edges("data_quality", check_data_quality)
    workflow.add_conditional_edges("feature_selection", check_features)
    workflow.add_conditional_edges("model_selection", check_analysis)
    workflow.add_edge("recommendation", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()


# ========== MODULAR ANALYSIS STRATEGIES ==========
class AnalysisStrategy:
    """Base class for swappable analysis strategies"""
    def analyze(self, df: pd.DataFrame, features: List[str]) -> Dict:
        raise NotImplementedError


class RegressionStrategy(AnalysisStrategy):
    """Linear regression approach"""
    def analyze(self, df, features):
        return _run_regression_analysis(df, features)


class CorrelationStrategy(AnalysisStrategy):
    """Simple correlation approach"""
    def analyze(self, df, features):
        return _run_correlation_analysis(df, features)


class DecisionTreeStrategy(AnalysisStrategy):
    """Decision tree approach (for non-linear relationships)"""
    def analyze(self, df: pd.DataFrame, features: List[str]) -> Dict:
        target = 'fastest_time'
        X = df[features].fillna(df[features].mean())
        y = df[target]

        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)

        # Feature importance from tree
        impacts = {feat: float(imp) for feat, imp in zip(features, model.feature_importances_)}
        sorted_impacts = sorted(impacts.items(), key=lambda x: x[1], reverse=True)

        return {
            'method': 'decision_tree',
            'impacts': impacts,
            'sorted_impacts': sorted_impacts,
            'model_score': float(model.score(X, y))
        }


# Export agents for use in demo
analysis_agent = model_selection_agent
engineer_agent = recommendation_agent
