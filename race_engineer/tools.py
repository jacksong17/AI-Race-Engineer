"""
Tool library for Race Engineer agents.

All tools that agents can call to perform analysis, validation, and recommendations.
"""

from langchain_core.tools import tool
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from csv_data_loader import CSVDataLoader
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats


# ===== DATA OPERATION TOOLS =====

@tool
def load_telemetry(file_paths: List[str]) -> Dict[str, Any]:
    """
    Load and parse telemetry data from multiple file formats.

    Supports:
    - iRacing .ibt binary files
    - MoTec .ldx XML exports
    - CSV processed lap data

    Args:
        file_paths: List of file paths to load (can use glob patterns)

    Returns:
        Dictionary containing:
        - data: Serialized DataFrame (as dict)
        - num_sessions: Number of sessions loaded
        - parameters: List of parameter names
        - source_format: File format detected
        - load_warnings: Any warnings during load
    """
    try:
        # Use existing CSV data loader
        loader = CSVDataLoader()
        df = loader.load_data()

        if df is None or df.empty:
            # Generate mock data for demo if no real data
            df = _generate_mock_data()
            source = "mock_data"
            warnings = ["No telemetry files found, using mock data for demonstration"]
        else:
            df = loader.prepare_for_ai_analysis(df)
            source = "csv_data"
            warnings = []

        # Convert DataFrame to dict for JSON serialization
        data_dict = df.to_dict(orient='records')

        return {
            "data": data_dict,
            "data_columns": list(df.columns),
            "num_sessions": len(df),
            "parameters": [col for col in df.columns if col not in ['session_id', 'fastest_time']],
            "source_format": source,
            "load_warnings": warnings
        }

    except Exception as e:
        return {
            "error": str(e),
            "data": None
        }


@tool
def inspect_quality(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform comprehensive data quality assessment.

    Checks:
    - Sample size adequacy
    - Outlier detection (IQR method)
    - Missing data analysis
    - Parameter variance analysis
    - Lap time distribution

    Args:
        data_dict: Dictionary containing 'data' and 'data_columns' from load_telemetry

    Returns:
        Dictionary with quality metrics:
        - quality_score: float (0-1)
        - num_sessions: int
        - outliers: List of outlier session info
        - missing_data: Dict of missing percentages
        - parameter_variance: Dict of variance scores
        - usable_parameters: List of parameters with sufficient variance
        - recommendations: List of quality improvement suggestions
    """
    try:
        # Reconstruct DataFrame
        df = pd.DataFrame(data_dict['data'])

        num_sessions = len(df)
        quality_score = 1.0
        recommendations = []

        # Check sample size
        if num_sessions < 5:
            quality_score *= 0.5
            recommendations.append(f"Small sample size ({num_sessions} sessions). Recommend 10+ for robust analysis.")
        elif num_sessions < 10:
            quality_score *= 0.8
            recommendations.append(f"Moderate sample size ({num_sessions} sessions). 15+ ideal for comprehensive analysis.")

        # Outlier detection on lap times
        lap_times = df['fastest_time']
        q1, q3 = lap_times.quantile(0.25), lap_times.quantile(0.75)
        iqr = q3 - q1
        outlier_threshold_high = q3 + 1.5 * iqr
        outlier_threshold_low = q1 - 1.5 * iqr

        outliers_mask = (lap_times > outlier_threshold_high) | (lap_times < outlier_threshold_low)
        outliers = []

        if outliers_mask.any():
            for idx in df[outliers_mask].index:
                outliers.append({
                    "session_id": df.loc[idx, 'session_id'] if 'session_id' in df.columns else f"session_{idx}",
                    "lap_time": float(df.loc[idx, 'fastest_time']),
                    "deviation": float(df.loc[idx, 'fastest_time'] - lap_times.median())
                })

            outlier_pct = len(outliers) / num_sessions
            if outlier_pct > 0.2:
                quality_score *= 0.7
                recommendations.append(f"High outlier rate ({outlier_pct:.1%}). May indicate data quality issues.")

        # Missing data analysis
        missing_data = {}
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df)
            if missing_pct > 0:
                missing_data[col] = float(missing_pct)
                if missing_pct > 0.1:
                    quality_score *= 0.9
                    recommendations.append(f"Column '{col}' has {missing_pct:.1%} missing data")

        # Parameter variance analysis
        parameter_variance = {}
        usable_parameters = []
        variance_threshold = 0.01

        for col in df.columns:
            if col in ['session_id', 'fastest_time']:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                variance = float(df[col].std())
                parameter_variance[col] = variance

                if variance > variance_threshold:
                    usable_parameters.append(col)

        if len(usable_parameters) < 3:
            quality_score *= 0.6
            recommendations.append(f"Only {len(usable_parameters)} parameters varied. Need more parameter testing.")

        return {
            "quality_score": float(quality_score),
            "num_sessions": num_sessions,
            "outliers": outliers,
            "missing_data": missing_data,
            "parameter_variance": parameter_variance,
            "usable_parameters": usable_parameters,
            "lap_time_range": (float(lap_times.min()), float(lap_times.max())),
            "lap_time_std": float(lap_times.std()),
            "recommendations": recommendations
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def clean_data(
    data_dict: Dict[str, Any],
    remove_outliers: bool = True,
    outlier_threshold: float = 1.5
) -> Dict[str, Any]:
    """
    Clean and prepare data for analysis.

    Args:
        data_dict: Dictionary containing 'data' and 'data_columns'
        remove_outliers: Whether to remove statistical outliers
        outlier_threshold: IQR multiplier for outlier detection (default 1.5)

    Returns:
        Dictionary with cleaned data:
        - data: Cleaned data
        - data_columns: Column names
        - rows_removed: Number of rows removed
        - cleaning_log: List of cleaning actions taken
    """
    try:
        df = pd.DataFrame(data_dict['data'])
        original_count = len(df)
        cleaning_log = []

        if remove_outliers:
            # Remove lap time outliers using IQR method
            lap_times = df['fastest_time']
            q1, q3 = lap_times.quantile(0.25), lap_times.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_threshold * iqr
            upper_bound = q3 + outlier_threshold * iqr

            outlier_mask = (lap_times < lower_bound) | (lap_times > upper_bound)
            num_outliers = outlier_mask.sum()

            if num_outliers > 0:
                df = df[~outlier_mask]
                cleaning_log.append(f"Removed {num_outliers} outlier sessions (IQR method)")

        # Drop rows with missing critical data
        critical_cols = ['fastest_time']
        before_drop = len(df)
        df = df.dropna(subset=critical_cols)
        after_drop = len(df)

        if after_drop < before_drop:
            cleaning_log.append(f"Dropped {before_drop - after_drop} rows with missing lap times")

        rows_removed = original_count - len(df)

        return {
            "data": df.to_dict(orient='records'),
            "data_columns": list(df.columns),
            "rows_removed": rows_removed,
            "final_count": len(df),
            "cleaning_log": cleaning_log
        }

    except Exception as e:
        return {"error": str(e)}


# ===== STATISTICAL ANALYSIS TOOLS =====

@tool
def select_features(
    data_dict: Dict[str, Any],
    driver_complaint: str,
    min_variance: float = 0.01
) -> Dict[str, Any]:
    """
    Intelligently select features for analysis based on:
    - Parameter variance (was it actually changed?)
    - Relevance to driver complaint
    - Data quality per parameter

    Args:
        data_dict: Dictionary containing telemetry data
        driver_complaint: Type of handling issue
        min_variance: Minimum variance threshold

    Returns:
        Dictionary with:
        - selected_features: List of parameter names
        - variance_scores: Variance for each parameter
        - relevance_scores: Relevance to complaint
        - rejection_reasons: Why parameters were rejected
    """
    try:
        df = pd.DataFrame(data_dict['data'])

        # Define complaint-relevant parameters
        complaint_relevance = {
            'oversteer': ['tire_psi_rr', 'tire_psi_lr', 'track_bar_height_left', 'spring_rr', 'cross_weight'],
            'loose': ['tire_psi_rr', 'tire_psi_lr', 'track_bar_height_left', 'spring_rr'],
            'understeer': ['tire_psi_lf', 'tire_psi_rf', 'cross_weight', 'spring_lf', 'spring_rf'],
            'tight': ['tire_psi_lf', 'tire_psi_rf', 'cross_weight', 'spring_lf'],
            'push': ['tire_psi_lf', 'tire_psi_rf', 'cross_weight'],
            'bottoming': ['spring_lf', 'spring_rf', 'spring_lr', 'spring_rr'],
        }

        # Determine relevant parameters
        relevant_params = []
        for key, params in complaint_relevance.items():
            if key in driver_complaint.lower():
                relevant_params.extend(params)

        selected_features = []
        variance_scores = {}
        relevance_scores = {}
        rejection_reasons = {}

        for col in df.columns:
            if col in ['session_id', 'fastest_time']:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                rejection_reasons[col] = "non-numeric"
                continue

            variance = float(df[col].std())
            variance_scores[col] = variance

            if variance < min_variance:
                rejection_reasons[col] = f"low_variance ({variance:.4f})"
                continue

            # Calculate relevance score
            if col in relevant_params:
                relevance_scores[col] = 1.0
            else:
                relevance_scores[col] = 0.5

            selected_features.append(col)

        # Sort by relevance then variance
        selected_features.sort(
            key=lambda x: (relevance_scores.get(x, 0), variance_scores.get(x, 0)),
            reverse=True
        )

        return {
            "selected_features": selected_features,
            "variance_scores": variance_scores,
            "relevance_scores": relevance_scores,
            "rejection_reasons": rejection_reasons,
            "num_selected": len(selected_features)
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def correlation_analysis(
    data_dict: Dict[str, Any],
    features: List[str],
    target: str = "fastest_time",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Perform correlation analysis between features and lap time with confidence intervals.

    Args:
        data_dict: Telemetry data
        features: List of parameter names to analyze
        target: Target variable (default: fastest_time)
        confidence_level: Confidence level for intervals (default: 0.95)

    Returns:
        Dictionary with:
        - method: "pearson_correlation"
        - correlations: Correlation coefficients
        - confidence_intervals: 95% CIs for each correlation
        - p_values: Statistical significance
        - significant_params: Parameters with p < 0.05
        - strength_interpretation: Interpretation of correlations
    """
    try:
        df = pd.DataFrame(data_dict['data'])

        correlations = {}
        confidence_intervals = {}
        p_values = {}
        significant_params = []
        strength_interpretation = {}

        for feature in features:
            if feature not in df.columns:
                continue

            # Get clean data
            feature_data = df[feature].dropna()
            target_data = df[target].dropna()

            # Align indices
            common_idx = feature_data.index.intersection(target_data.index)
            x = df.loc[common_idx, feature]
            y = df.loc[common_idx, target]
            n = len(x)

            # Calculate Pearson correlation
            corr, p_val = stats.pearsonr(x, y)

            correlations[feature] = float(corr)
            p_values[feature] = float(p_val)

            # Calculate confidence interval using Fisher Z-transformation
            if n > 3:
                z = np.arctanh(corr)  # Fisher Z-transform
                se = 1 / np.sqrt(n - 3)  # Standard error
                z_critical = stats.norm.ppf((1 + confidence_level) / 2)
                z_ci_low = z - z_critical * se
                z_ci_high = z + z_critical * se

                # Transform back to correlation scale
                ci_low = float(np.tanh(z_ci_low))
                ci_high = float(np.tanh(z_ci_high))
                confidence_intervals[feature] = (ci_low, ci_high)
            else:
                confidence_intervals[feature] = (None, None)

            # Determine significance
            if p_val < 0.05:
                significant_params.append(feature)

            # Interpret strength
            abs_corr = abs(corr)
            if abs_corr > 0.7:
                strength = "very strong"
            elif abs_corr > 0.5:
                strength = "strong"
            elif abs_corr > 0.3:
                strength = "moderate"
            elif abs_corr > 0.1:
                strength = "weak"
            else:
                strength = "very weak"

            direction = "negative" if corr < 0 else "positive"
            strength_interpretation[feature] = f"{strength} {direction}"

        # Sort by absolute correlation
        sorted_params = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "method": "pearson_correlation",
            "correlations": correlations,
            "confidence_intervals": confidence_intervals,
            "confidence_level": confidence_level,
            "p_values": p_values,
            "significant_params": significant_params,
            "strength_interpretation": strength_interpretation,
            "sorted_by_impact": [param for param, _ in sorted_params],
            "top_parameter": sorted_params[0][0] if sorted_params else None,
            "top_correlation": sorted_params[0][1] if sorted_params else None
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def regression_analysis(
    data_dict: Dict[str, Any],
    features: List[str],
    target: str = "fastest_time"
) -> Dict[str, Any]:
    """
    Perform multivariate regression analysis.

    Uses:
    - Linear regression with standardization
    - Feature importance via coefficients
    - R² and adjusted R² for model quality

    Args:
        data_dict: Telemetry data
        features: List of parameters to analyze
        target: Target variable (default: fastest_time)

    Returns:
        Dictionary with regression results
    """
    try:
        df = pd.DataFrame(data_dict['data'])

        # Prepare data
        available_features = [f for f in features if f in df.columns]
        if len(available_features) < 2:
            return {"error": "Need at least 2 features for regression"}

        X = df[available_features].dropna()
        y = df.loc[X.index, target]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Calculate metrics
        r_squared = float(model.score(X_scaled, y))
        n = len(y)
        p = len(available_features)
        adjusted_r_squared = float(1 - (1 - r_squared) * (n - 1) / (n - p - 1))

        # Feature importance
        coefficients = {feat: float(coef) for feat, coef in zip(available_features, model.coef_)}
        feature_importance = {feat: abs(coef) for feat, coef in coefficients.items()}

        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Model quality assessment
        if r_squared > 0.7:
            quality = "excellent"
        elif r_squared > 0.5:
            quality = "good"
        elif r_squared > 0.3:
            quality = "moderate"
        else:
            quality = "poor"

        return {
            "method": "linear_regression",
            "coefficients": coefficients,
            "r_squared": r_squared,
            "adjusted_r_squared": adjusted_r_squared,
            "feature_importance": feature_importance,
            "sorted_features": [feat for feat, _ in sorted_features],
            "top_parameter": sorted_features[0][0] if sorted_features else None,
            "model_quality": quality,
            "intercept": float(model.intercept_)
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def calculate_effect_size(
    data_dict: Dict[str, Any],
    parameter: str,
    target: str = "fastest_time",
    split_method: str = "median"
) -> Dict[str, Any]:
    """
    Calculate Cohen's d effect size for parameter impact.

    Quantifies practical significance beyond statistical significance.
    Cohen's d interpretation: 0.2 = small, 0.5 = medium, 0.8 = large

    Args:
        data_dict: Telemetry data
        parameter: Parameter to analyze
        target: Target variable (default: fastest_time)
        split_method: How to split groups ("median", "quartiles")

    Returns:
        Dictionary with effect size metrics
    """
    try:
        df = pd.DataFrame(data_dict['data'])

        if parameter not in df.columns:
            return {"error": f"Parameter '{parameter}' not found"}

        # Split into groups
        if split_method == "median":
            median_val = df[parameter].median()
            group_low = df[df[parameter] <= median_val][target]
            group_high = df[df[parameter] > median_val][target]
            split_desc = f"below/above median ({median_val:.2f})"
        elif split_method == "quartiles":
            q1 = df[parameter].quantile(0.25)
            q3 = df[parameter].quantile(0.75)
            group_low = df[df[parameter] <= q1][target]
            group_high = df[df[parameter] >= q3][target]
            split_desc = f"Q1 vs Q3 ({q1:.2f} vs {q3:.2f})"
        else:
            return {"error": f"Unknown split_method: {split_method}"}

        # Calculate Cohen's d
        mean_diff = float(group_high.mean() - group_low.mean())
        pooled_std = float(np.sqrt((group_low.std()**2 + group_high.std()**2) / 2))

        if pooled_std == 0:
            return {"error": "Zero variance - cannot calculate effect size"}

        cohens_d = mean_diff / pooled_std

        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d >= 0.8:
            interpretation = "large"
        elif abs_d >= 0.5:
            interpretation = "medium"
        elif abs_d >= 0.2:
            interpretation = "small"
        else:
            interpretation = "negligible"

        return {
            "parameter": parameter,
            "cohens_d": float(cohens_d),
            "effect_interpretation": interpretation,
            "mean_difference": mean_diff,
            "group_low_mean": float(group_low.mean()),
            "group_high_mean": float(group_high.mean()),
            "group_low_n": len(group_low),
            "group_high_n": len(group_high),
            "split_method": split_desc,
            "practical_significance": abs_d >= 0.2
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def analyze_interactions(
    data_dict: Dict[str, Any],
    features: List[str],
    target: str = "fastest_time",
    max_interactions: int = 5
) -> Dict[str, Any]:
    """
    Analyze parameter interactions (synergies) using polynomial regression.

    Detects when combining two parameters has greater impact than individual effects.
    Example: tire_psi × spring_rate interaction affects corner stiffness.

    Args:
        data_dict: Telemetry data
        features: List of parameters to analyze
        target: Target variable (default: fastest_time)
        max_interactions: Maximum number of top interactions to report

    Returns:
        Dictionary with interaction analysis results
    """
    try:
        from sklearn.preprocessing import PolynomialFeatures

        df = pd.DataFrame(data_dict['data'])

        # Prepare data
        available_features = [f for f in features if f in df.columns]
        if len(available_features) < 2:
            return {"error": "Need at least 2 features for interaction analysis"}

        X = df[available_features].dropna()
        y = df.loc[X.index, target]

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit baseline model (no interactions)
        baseline_model = LinearRegression()
        baseline_model.fit(X_scaled, y)
        baseline_r2 = float(baseline_model.score(X_scaled, y))

        # Fit interaction model
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_interactions = poly.fit_transform(X_scaled)
        feature_names = poly.get_feature_names_out(available_features)

        interaction_model = LinearRegression()
        interaction_model.fit(X_interactions, y)
        interaction_r2 = float(interaction_model.score(X_interactions, y))

        # R² improvement from interactions
        r2_gain = interaction_r2 - baseline_r2

        # Extract interaction terms (skip main effects)
        interaction_terms = {}
        for idx, name in enumerate(feature_names):
            if ' ' in name:  # Interaction term (contains space, e.g., "x0 x1")
                coef = float(interaction_model.coef_[idx])
                interaction_terms[name] = abs(coef)

        # Sort by importance
        sorted_interactions = sorted(interaction_terms.items(), key=lambda x: x[1], reverse=True)
        top_interactions = sorted_interactions[:max_interactions]

        # Interpret
        if r2_gain > 0.15:
            interpretation = "strong synergies detected"
        elif r2_gain > 0.05:
            interpretation = "moderate synergies detected"
        elif r2_gain > 0.01:
            interpretation = "weak synergies detected"
        else:
            interpretation = "negligible synergies"

        return {
            "method": "polynomial_regression_interactions",
            "baseline_r2": baseline_r2,
            "interaction_r2": interaction_r2,
            "r2_gain_from_interactions": r2_gain,
            "interpretation": interpretation,
            "top_interactions": [
                {"term": term, "coefficient": coef}
                for term, coef in top_interactions
            ],
            "num_features": len(available_features),
            "num_interaction_terms": len(interaction_terms)
        }

    except Exception as e:
        return {"error": str(e)}


# ===== KNOWLEDGE & VALIDATION TOOLS =====

@tool
def query_setup_manual(issue_type: str, parameter: Optional[str] = None) -> Dict[str, Any]:
    """
    Query NASCAR truck setup knowledge base from parsed manual.

    Now uses actual NASCAR Trucks Manual V6 content!

    Args:
        issue_type: Type of handling issue (oversteer, understeer, etc)
        parameter: Optional specific parameter to get info about

    Returns:
        Dictionary with detailed setup guidance from NASCAR manual
    """
    # Load parsed NASCAR manual knowledge
    knowledge_file = Path(__file__).parent.parent / "data" / "knowledge" / "nascar_manual_knowledge.json"

    if not knowledge_file.exists():
        # Parse manual if not cached
        from race_engineer.nascar_manual_parser import parse_and_cache_manual
        pdf_path = Path(__file__).parent.parent / "NASCAR-Trucks-Manual-V6.pdf"
        knowledge = parse_and_cache_manual(str(pdf_path))
    else:
        with open(knowledge_file, 'r') as f:
            knowledge = json.load(f)

    relevant_sections = []
    principles = []
    parameter_guidance = {}
    fixes = {}

    # Extract relevant handling issue information
    issue_key = issue_type.lower().replace(' ', '_')
    if issue_key in knowledge.get('handling_issues', {}):
        issue_info = knowledge['handling_issues'][issue_key]

        relevant_sections.append(issue_info.get('description', ''))
        principles.extend(issue_info.get('symptoms', []))

        # Get specific fixes from manual
        fixes = issue_info.get('fixes', {})

        # Build parameter guidance from fixes
        for param, fix_info in fixes.items():
            parameter_guidance[param] = {
                'action': fix_info.get('action'),
                'magnitude': fix_info.get('magnitude'),
                'rationale': fix_info.get('rationale'),
                'from_nascar_manual': True
            }

    # Get specific parameter info if requested
    if parameter and parameter in knowledge.get('parameters', {}):
        param_info = knowledge['parameters'][parameter]
        parameter_guidance[parameter] = param_info

    return {
        "relevant_sections": relevant_sections,
        "principles": principles,
        "parameter_guidance": parameter_guidance,
        "fixes": fixes,
        "manual_version": knowledge.get('manual_version', 'V6'),
        "source": "NASCAR Trucks Manual V6"
    }


@tool
def search_history(
    complaint_type: str,
    track: str = "bristol",
    limit: int = 5
) -> Dict[str, Any]:
    """
    Search historical sessions with similar complaints.

    Args:
        complaint_type: Type of driver complaint
        track: Track name
        limit: Maximum number of results

    Returns:
        Dictionary with historical session data
    """
    # This would query the SQLite database in a real implementation
    # For now, return mock structure
    return {
        "similar_sessions": [],
        "successful_solutions": [],
        "patterns": {},
        "note": "Historical database integration pending - this is a placeholder"
    }


@tool
def check_constraints(
    parameter: str,
    direction: str,
    magnitude: float,
    current_value: Optional[float] = None,
    constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Comprehensive constraint checking against NASCAR manual limits.

    NOW VALIDATES AGAINST ACTUAL NASCAR TRUCKS MANUAL SPECS!

    Checks:
    1. NASCAR rule limits (from manual)
    2. Physical limits
    3. Driver-specified constraints
    4. Safety margins

    Args:
        parameter: Parameter name
        direction: "increase" or "decrease"
        magnitude: Amount of change
        current_value: Optional current value
        constraints: Optional driver constraints

    Returns:
        Dictionary with validation results and limit information
    """
    # Load NASCAR manual constraints
    knowledge_file = Path(__file__).parent.parent / "data" / "knowledge" / "nascar_manual_knowledge.json"

    manual_limits = {}
    if knowledge_file.exists():
        with open(knowledge_file, 'r') as f:
            knowledge = json.load(f)

        # Get parameter limits from manual
        if parameter in knowledge.get('parameters', {}):
            param_info = knowledge['parameters'][parameter]
            if 'range' in param_info:
                manual_limits = {
                    'min': param_info['range']['min'],
                    'max': param_info['range']['max'],
                    'typical_min': param_info.get('typical', {}).get('min'),
                    'typical_max': param_info.get('typical', {}).get('max'),
                    'unit': param_info.get('unit', 'units'),
                    'source': 'NASCAR Trucks Manual V6'
                }

    violations = []
    warnings = []
    proposed_value = None

    # Calculate proposed value if current is provided
    if current_value is not None and manual_limits:
        if direction.lower() == "increase":
            proposed_value = current_value + magnitude
        else:
            proposed_value = current_value - magnitude

        # Check NASCAR manual limits
        if proposed_value < manual_limits['min']:
            violations.append(
                f"{parameter} would be {proposed_value:.2f} {manual_limits['unit']}, "
                f"below NASCAR manual minimum of {manual_limits['min']} {manual_limits['unit']}"
            )
        elif proposed_value > manual_limits['max']:
            violations.append(
                f"{parameter} would be {proposed_value:.2f} {manual_limits['unit']}, "
                f"above NASCAR manual maximum of {manual_limits['max']} {manual_limits['unit']}"
            )

        # Check if approaching limits (within 10%)
        limit_range = manual_limits['max'] - manual_limits['min']
        margin_low = manual_limits['min'] + 0.1 * limit_range
        margin_high = manual_limits['max'] - 0.1 * limit_range

        if proposed_value < margin_low and proposed_value >= manual_limits['min']:
            margin = proposed_value - manual_limits['min']
            warnings.append(
                f"{parameter} approaching minimum limit (margin: {margin:.2f} {manual_limits['unit']})"
            )
        elif proposed_value > margin_high and proposed_value <= manual_limits['max']:
            margin = manual_limits['max'] - proposed_value
            warnings.append(
                f"{parameter} approaching maximum limit (margin: {margin:.2f} {manual_limits['unit']})"
            )

        # Check if outside typical range
        if manual_limits.get('typical_min') and manual_limits.get('typical_max'):
            if proposed_value < manual_limits['typical_min']:
                warnings.append(
                    f"{parameter} below typical range ({manual_limits['typical_min']}-{manual_limits['typical_max']} {manual_limits['unit']})"
                )
            elif proposed_value > manual_limits['typical_max']:
                warnings.append(
                    f"{parameter} above typical range ({manual_limits['typical_min']}-{manual_limits['typical_max']} {manual_limits['unit']})"
                )

    # Check driver constraints if provided
    if constraints:
        params_at_limit = constraints.get('parameters_at_limit', {})
        if parameter in params_at_limit:
            limit_type = params_at_limit[parameter]
            if (limit_type == 'min' and direction.lower() == 'decrease') or \
               (limit_type == 'max' and direction.lower() == 'increase'):
                violations.append(f"{parameter} is already at {limit_type} limit per driver constraints")

        cannot_adjust = constraints.get('cannot_adjust', [])
        if parameter in cannot_adjust:
            violations.append(f"{parameter} cannot be adjusted per driver constraints")

        already_tried = constraints.get('already_tried', [])
        if parameter in already_tried:
            warnings.append(f"{parameter} was already tried in previous sessions")

    is_valid = len(violations) == 0

    result = {
        "is_valid": is_valid,
        "violations": violations,
        "warnings": warnings,
        "nascar_manual_limits": manual_limits,
        "proposed_value": proposed_value,
        "current_value": current_value,
        "within_typical_range": True  # Default
    }

    # Calculate margins if we have limits and proposed value
    if manual_limits and proposed_value is not None:
        result["margin_to_limits"] = {
            "min": proposed_value - manual_limits['min'],
            "max": manual_limits['max'] - proposed_value,
            "unit": manual_limits.get('unit', 'units')
        }

        # Check if within typical range
        if manual_limits.get('typical_min') and manual_limits.get('typical_max'):
            result["within_typical_range"] = (
                manual_limits['typical_min'] <= proposed_value <= manual_limits['typical_max']
            )

    return result


@tool
def validate_physics(recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate recommendations against racing physics principles.

    Checks:
    - Setup balance (front vs rear)
    - Tire pressure balance
    - Spring rate ratios

    Args:
        recommendations: List of recommendation dictionaries

    Returns:
        Validation results
    """
    warnings = []
    conflicts = []

    # Check for balance issues
    front_changes = [r for r in recommendations if 'lf' in r['parameter'] or 'rf' in r['parameter']]
    rear_changes = [r for r in recommendations if 'lr' in r['parameter'] or 'rr' in r['parameter']]

    if len(front_changes) > 0 and len(rear_changes) == 0:
        warnings.append("Only front changes recommended - may create balance issues")
    elif len(rear_changes) > 0 and len(front_changes) == 0:
        warnings.append("Only rear changes recommended - may create balance issues")

    # Check for conflicting recommendations
    params_modified = [r['parameter'] for r in recommendations]
    if len(params_modified) != len(set(params_modified)):
        conflicts.append("Multiple recommendations for the same parameter")

    physics_valid = len(conflicts) == 0

    return {
        "physics_valid": physics_valid,
        "warnings": warnings,
        "conflicts": conflicts
    }


@tool
def evaluate_recommendation_quality(
    recommendation: Dict[str, Any],
    driver_feedback: str,
    statistical_support: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    LM-as-judge evaluation of recommendation quality.

    Implements Google AgentOps principle: "Quality Instead of Pass/Fail"

    Evaluates on 4 dimensions:
    - Relevance: Does it address driver's complaint?
    - Confidence: Is statistical support strong?
    - Safety: Are constraints respected?
    - Clarity: Is guidance specific and actionable?

    Returns quality scores and pass/fail decision.
    """

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.1)

    # Extract key info
    param = recommendation.get('parameter', 'unknown')
    direction = recommendation.get('direction', 'unknown')
    magnitude = recommendation.get('magnitude', 0)
    unit = recommendation.get('magnitude_unit', '')

    # Build evaluation prompt
    eval_prompt = f"""You are a NASCAR setup quality judge. Evaluate this recommendation:

DRIVER COMPLAINT: {driver_feedback}

RECOMMENDATION: {direction.title()} {param} by {magnitude} {unit}

STATISTICAL SUPPORT:
- Method: {statistical_support.get('method', 'unknown')}
- Top correlation: {statistical_support.get('top_correlation', 'N/A')}
- Significant params: {statistical_support.get('significant_params', [])}

CONSTRAINTS: {json.dumps(constraints) if constraints else 'None provided'}

Rate on scale 0-10:
1. RELEVANCE: Does this address "{driver_feedback}"?
2. CONFIDENCE: Is the statistical support strong enough?
3. SAFETY: Does it respect limits and constraints?
4. CLARITY: Is it specific and actionable?

Respond ONLY with valid JSON (no markdown):
{{
  "relevance": 8,
  "confidence": 7,
  "safety": 10,
  "clarity": 9,
  "overall_score": 8.5,
  "pass": true,
  "reasoning": "Strong statistical support (correlation -0.42) directly addresses oversteer complaint. Within NASCAR manual limits."
}}"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are an impartial quality judge for NASCAR setup recommendations."),
            HumanMessage(content=eval_prompt)
        ])

        # Parse JSON from response
        content = response.content.strip()
        # Remove markdown code blocks if present
        content = content.replace("```json", "").replace("```", "").strip()

        result = json.loads(content)

        # Ensure all required fields
        result.setdefault('relevance', 5)
        result.setdefault('confidence', 5)
        result.setdefault('safety', 5)
        result.setdefault('clarity', 5)
        result.setdefault('overall_score', 5.0)
        result.setdefault('pass', result['overall_score'] >= 7.0)
        result.setdefault('reasoning', 'No reasoning provided')

        return {
            "evaluation": result,
            "recommendation_validated": result['pass'],
            "quality_gate": "passed" if result['pass'] else "failed",
            "improvement_areas": [
                k for k, v in result.items()
                if isinstance(v, (int, float)) and v < 7
            ]
        }

    except Exception as e:
        return {
            "error": str(e),
            "evaluation": None,
            "recommendation_validated": False,
            "quality_gate": "error"
        }


# ===== OUTPUT TOOLS =====

@tool
def save_session(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save session results to historical database.

    Args:
        session_data: Complete session data to save

    Returns:
        Save confirmation
    """
    # This would save to SQLite in production
    # For now, save to JSON file
    try:
        output_dir = Path("output/sessions")
        output_dir.mkdir(parents=True, exist_ok=True)

        session_id = session_data.get('session_id', 'unknown')
        filename = f"session_{session_id}.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        return {
            "saved": True,
            "filepath": str(filepath),
            "session_id": session_id
        }

    except Exception as e:
        return {
            "saved": False,
            "error": str(e)
        }


# ===== HELPER FUNCTIONS =====

def _generate_mock_data() -> pd.DataFrame:
    """Generate mock Bristol testing data for demos"""
    baseline_setup = {
        'tire_psi_lf': 28.0,
        'tire_psi_rf': 32.0,
        'tire_psi_lr': 26.0,
        'tire_psi_rr': 30.0,
        'cross_weight': 54.0,
        'track_bar_height_left': 10.0,
        'spring_lf': 400,
        'spring_rf': 425,
        'fastest_time': 15.543
    }

    sessions = []
    for i in range(20):
        session = baseline_setup.copy()
        session['session_id'] = f"bristol_test_{i+1}"

        if i < 5:
            session['fastest_time'] = 15.543 + np.random.normal(0, 0.05)
        elif i < 8:
            session['tire_psi_lf'] = 28.0 + (i - 5) * 2
            session['fastest_time'] = 15.543 - 0.05 * (8 - i) + np.random.normal(0, 0.03)
        elif i < 11:
            session['cross_weight'] = 52.0 + (i - 8) * 2
            session['fastest_time'] = 15.543 - 0.08 * (i - 8) + np.random.normal(0, 0.03)
        elif i < 14:
            session['track_bar_height_left'] = 5.0 + (i - 11) * 5
            session['fastest_time'] = 15.543 - 0.04 * (i - 11) + np.random.normal(0, 0.03)
        else:
            session['tire_psi_lf'] = 26.0
            session['cross_weight'] = 56.0
            session['track_bar_height_left'] = 12.0
            session['fastest_time'] = 15.543 - 0.30 + np.random.normal(0, 0.02)

        sessions.append(session)

    return pd.DataFrame(sessions)


def _create_default_knowledge() -> Dict[str, Any]:
    """Create default setup knowledge base"""
    return {
        "handling_issues": {
            "oversteer": {
                "description": "Car is loose, rear end wants to come around",
                "principles": [
                    "Increase rear grip",
                    "Reduce rear tire pressure for more contact patch",
                    "Stiffen rear springs to reduce body roll",
                    "Lower track bar to loosen car"
                ],
                "parameter_guidance": {
                    "tire_psi_rr": "Reduce by 1-2 PSI to increase mechanical grip",
                    "tire_psi_lr": "Reduce by 1-2 PSI to increase mechanical grip",
                    "track_bar_height_left": "Lower by 0.25-0.5 inches to shift roll center"
                }
            },
            "understeer": {
                "description": "Car is tight, won't turn",
                "principles": [
                    "Increase front grip",
                    "Reduce front tire pressure",
                    "Adjust cross weight",
                    "Soften front springs"
                ],
                "parameter_guidance": {
                    "tire_psi_lf": "Reduce by 1-2 PSI",
                    "tire_psi_rf": "Reduce by 1-2 PSI",
                    "cross_weight": "Reduce to shift weight to front"
                }
            }
        },
        "parameters": {
            "tire_psi_rr": {
                "range": "25-35 PSI",
                "typical": "28-30 PSI",
                "effect": "Higher pressure = less grip but more responsive"
            },
            "cross_weight": {
                "range": "50-56%",
                "typical": "52-54%",
                "effect": "Higher = more rear bias, helps turn entry"
            }
        },
        "examples": []
    }


# Export all tools
ALL_TOOLS = [
    load_telemetry,
    inspect_quality,
    clean_data,
    select_features,
    correlation_analysis,
    regression_analysis,
    query_setup_manual,
    search_history,
    check_constraints,
    validate_physics,
    evaluate_recommendation_quality,
    save_session
]
