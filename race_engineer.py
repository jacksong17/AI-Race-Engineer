# race_engineer.py
"""
AI Race Engineer with Real Agent Decision-Making
Each agent makes visible, context-aware decisions
"""

import pandas as pd
from typing import TypedDict, List, Dict, Optional
from pathlib import Path
from langgraph.graph import StateGraph, END

# Import tools for analysis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# State with decision tracking
class RaceEngineerState(TypedDict):
    raw_setup_data: Optional[pd.DataFrame]
    driver_feedback: Optional[Dict]
    driver_diagnosis: Optional[Dict]  # Agent 1's interpretation
    data_quality_decision: Optional[str]
    analysis_strategy: Optional[str]
    selected_features: Optional[List[str]]
    analysis: Optional[Dict]
    recommendation: Optional[str]
    error: Optional[str]

    # Session Memory Fields (for iterative learning across stints)
    session_history: Optional[List[Dict]]  # Previous sessions for context
    session_timestamp: Optional[str]  # ISO timestamp for this session
    learning_metrics: Optional[Dict]  # Aggregated patterns across sessions
    previous_recommendations: Optional[List]  # Last 3-5 recommendations
    outcome_feedback: Optional[Dict]  # User validation of recommendation
    convergence_progress: Optional[float]  # % improvement trend

    # Driver Context Fields (preserve all driver input)
    raw_driver_feedback: Optional[str]  # Original driver message (unprocessed)
    driver_constraints: Optional[Dict]  # Extracted limits and constraints
    setup_knowledge_base: Optional[Dict]  # Car setup manual context


# Quiet mode helper
def _is_quiet_mode():
    """Check if running in quiet mode (minimal output)"""
    import os
    return os.environ.get('QUIET_MODE') == '1'

# State visibility helper
def _print_state_transition(from_agent: str, to_agent: str, state: RaceEngineerState):
    """Display state being passed between agents for demo visibility"""
    if _is_quiet_mode():
        return  # Suppress state handoffs in quiet mode

    print(f"\n{'='*70}")
    print(f"  STATE HANDOFF: {from_agent} -> {to_agent}")
    print(f"{'='*70}")

    # Show relevant state fields (exclude raw data)
    if state.get('driver_diagnosis'):
        diag = state['driver_diagnosis']
        print(f"   driver_diagnosis:")
        print(f"      diagnosis: {diag.get('diagnosis', 'N/A')}")
        print(f"      priority_features: {diag.get('priority_features', [])}")

    if state.get('driver_constraints'):
        constraints = state['driver_constraints']
        if constraints.get('parameter_limits'):
            print(f"   driver_constraints:")
            for limit in constraints['parameter_limits']:
                print(f"      {limit['param']}: {limit['limit_type']}")

    if state.get('data_quality_decision'):
        print(f"   data_quality_decision: {state['data_quality_decision']}")

    if state.get('analysis_strategy'):
        print(f"   analysis_strategy: {state['analysis_strategy']}")

    if state.get('selected_features'):
        print(f"   selected_features: {state['selected_features']}")

    if state.get('analysis'):
        analysis = state['analysis']
        print(f"   analysis:")
        print(f"      method: {analysis.get('method', 'N/A')}")
        if 'most_impactful' in analysis:
            param, impact = analysis['most_impactful']
            print(f"      most_impactful: {param} ({impact:+.3f})")

    if state.get('error'):
        print(f"   error: {state['error']}")

    print(f"{'='*70}\n")

# ========== AGENT 1: TELEMETRY CHIEF (Data Quality + Driver Feedback) ==========
def telemetry_agent(state: RaceEngineerState):
    """
    Agent 1: Data Quality Assessor + Driver Feedback Interpreter
    DECISIONS:
    - What is driver feeling? (interpret feedback)
    - Which setup areas relate to driver complaint?
    - Remove outliers or keep them?
    - Sufficient data for analysis?
    """
    if not _is_quiet_mode():
        print("\n[AGENT 1] Telemetry Chief: Interpreting driver feedback...")

    # DECISION 0: Interpret driver feedback (Perception -> Reasoning)
    driver_feedback = state.get('driver_feedback', {})
    driver_diagnosis = {}

    if driver_feedback:
        complaint = driver_feedback.get('complaint', '')
        phase = driver_feedback.get('phase', '')

        print(f"   [DRIVER] Complaint: '{complaint}' during {phase}")

        # Agent reasons about what setup parameters affect this handling characteristic
        if 'loose' in complaint or 'oversteer' in complaint:
            priority_features = ['tire_psi_rr', 'tire_psi_lr', 'track_bar_height_left', 'spring_rf', 'spring_rr']
            diagnosis = "Oversteer (loose rear end)"
            technical_cause = "Insufficient rear grip - likely rear tire pressure or rear spring rates"
            print(f"   DIAGNOSIS: {diagnosis}")
            print(f"      Technical assessment: {technical_cause}")
            print(f"   DECISION: Prioritize REAR GRIP parameters")
            print(f"      Priority features: {', '.join(priority_features[:3])}")
        elif 'tight' in complaint or 'understeer' in complaint or 'push' in complaint:
            priority_features = ['tire_psi_lf', 'tire_psi_rf', 'cross_weight', 'spring_lf', 'spring_rf']
            diagnosis = "Understeer (tight front end)"
            technical_cause = "Insufficient front grip - likely front tire pressure or weight distribution"
            print(f"   DIAGNOSIS: {diagnosis}")
            print(f"      Technical assessment: {technical_cause}")
            print(f"   DECISION: Prioritize FRONT GRIP parameters")
            print(f"      Priority features: {', '.join(priority_features[:3])}")
        elif 'bottoming' in complaint or 'hitting' in complaint:
            priority_features = ['spring_lf', 'spring_rf', 'spring_lr', 'spring_rr']
            diagnosis = "Suspension bottoming out"
            technical_cause = "Insufficient spring stiffness or ride height"
            print(f"   DIAGNOSIS: {diagnosis}")
            print(f"      Technical assessment: {technical_cause}")
            print(f"   DECISION: Prioritize SPRING RATES")
        else:
            priority_features = []
            diagnosis = "General optimization needed"
            technical_cause = "Analyze all parameters for correlation"
            print(f"   DIAGNOSIS: {diagnosis}")
            print(f"   DECISION: Broad analysis of all parameters")

        driver_diagnosis = {
            'diagnosis': diagnosis,
            'technical_cause': technical_cause,
            'priority_features': priority_features,
            'complaint_type': complaint
        }
    else:
        print("   [INFO] No driver feedback provided - proceeding with general analysis")
        driver_diagnosis = {
            'diagnosis': 'General optimization',
            'priority_features': [],
            'complaint_type': None
        }

    print()
    print("   [DATA] Assessing data quality...")

    df = state.get('raw_setup_data')
    if df is None or df.empty:
        return {"error": "No data provided"}

    # DECISION 1: Outlier detection
    lap_times = df['fastest_time']
    print(f"   [DATA] Dataset: {len(df)} sessions")
    print(f"   [STATS] Lap time range: {lap_times.min():.3f}s - {lap_times.max():.3f}s")
    print(f"   [STATS] Variance: {lap_times.std():.3f}s")

    # Calculate IQR for outlier detection
    q1, q3 = lap_times.quantile(0.25), lap_times.quantile(0.75)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    outliers = df[lap_times > outlier_threshold]

    # Agent makes decision about outliers
    if len(outliers) > 0:
        print(f"   [WARNING] Found {len(outliers)} outlier(s) > {outlier_threshold:.3f}s")
        if len(outliers) < len(df) * 0.2:  # Less than 20%
            df_clean = df[lap_times <= outlier_threshold]
            print(f"   DECISION: Removing {len(outliers)} outliers (keeping {len(df_clean)} sessions)")
            decision = f"removed_{len(outliers)}_outliers"
        else:
            print(f"   DECISION: Keeping all data (outliers represent >20% of dataset)")
            df_clean = df
            decision = "kept_all_data"
    else:
        print(f"   DECISION: No outliers detected, proceeding with all {len(df)} sessions")
        df_clean = df
        decision = "no_outliers_found"

    # DECISION 2: Sample size check
    if len(df_clean) < 5:
        return {"error": f"Insufficient data: only {len(df_clean)} valid sessions (need 5+)"}

    print(f"   [OK] Data quality check passed: {len(df_clean)} sessions ready for analysis")

    # Prepare state update
    updated_state = {
        "raw_setup_data": df_clean,
        "driver_diagnosis": driver_diagnosis,
        "data_quality_decision": decision
    }

    # Merge with existing state for visibility
    merged_state = {**state, **updated_state}
    _print_state_transition("AGENT 1: Telemetry Chief", "AGENT 2: Data Scientist", merged_state)

    return updated_state

# ========== AGENT 2: DATA SCIENTIST (Feature Selection + Model Strategy) ==========
def analysis_agent(state: RaceEngineerState):
    """
    Agent 2: Strategic Analyst
    DECISIONS:
    - Which features have enough variance to analyze?
    - Correlation or regression analysis?
    - Which model based on data characteristics?
    """
    print("\n[AGENT 2] Data Scientist: Selecting analysis strategy...")

    df = state.get('raw_setup_data')
    if df is None or df.empty:
        return {"error": "No data to analyze"}

    # Read driver diagnosis from Agent 1
    driver_diagnosis = state.get('driver_diagnosis', {})
    priority_features = driver_diagnosis.get('priority_features', [])

    # DECISION 1: Feature selection (dynamic based on variance + driver feedback)
    potential_features = [
        'tire_psi_lf', 'tire_psi_rf', 'tire_psi_lr', 'tire_psi_rr',
        'cross_weight', 'track_bar_height_left', 'spring_lf', 'spring_rf'
    ]

    if priority_features:
        print(f"   [TARGET] Agent 1 identified priority areas: {driver_diagnosis.get('diagnosis')}")
        print(f"      Focusing analysis on: {', '.join(priority_features[:3])}")
        print(f"    DECISION: Prioritize driver-feedback-relevant parameters")
        print()

    selected_features = []
    priority_selected = []
    print(f"   [CHECKING] Evaluating {len(potential_features)} potential features...")

    for feature in potential_features:
        if feature not in df.columns:
            continue

        variance = df[feature].std()
        is_priority = feature in priority_features

        # Agent decides: only use features that actually varied during testing
        if variance > 0.01:  # Feature was changed meaningfully
            selected_features.append(feature)
            if is_priority:
                priority_selected.append(feature)
                print(f"       {feature:25s} (varied: stdev={variance:.2f}) [PRIORITY]")
            else:
                print(f"       {feature:25s} (varied: stdev={variance:.2f})")
        else:
            marker = "[PRIORITY - no variance]" if is_priority else ""
            print(f"      [X] {feature:25s} (constant: stdev={variance:.4f}) {marker}")

    if len(selected_features) < 2:
        return {"error": f"Insufficient variable features: only {len(selected_features)} found"}

    if priority_selected:
        print(f"    DECISION: Using {len(selected_features)} features ({len(priority_selected)} priority features identified)")
        print(f"      Priority features with variance: {', '.join(priority_selected)}")
    else:
        print(f"    DECISION: Using {len(selected_features)} features for analysis")

    # DECISION 2: Choose analysis strategy
    sample_size = len(df)
    feature_count = len(selected_features)
    variance = df['fastest_time'].std()

    print(f"\n   [ANALYSIS] Evaluating strategy options...")
    print(f"      • Sample size: {sample_size}")
    print(f"      • Feature count: {feature_count}")
    print(f"      • Lap time variance: {variance:.3f}s")

    # Agent makes strategic decision
    if sample_size < 10:
        strategy = "correlation"
        reason = "small sample size"
    elif feature_count > sample_size / 2:
        strategy = "correlation"
        reason = "high feature-to-sample ratio"
    elif variance < 0.15:
        strategy = "correlation"
        reason = "low lap time variance"
    else:
        strategy = "regression"
        reason = "adequate data for regression"

    print(f"    DECISION: Using {strategy.upper()} analysis ({reason})")

    # Execute chosen strategy
    target = 'fastest_time'

    try:
        if strategy == "correlation":
            # Simple correlation analysis
            impacts = {}
            print(f"\n   [STATS] Running correlation analysis...")
            for feature in selected_features:
                corr = df[[feature, target]].corr().iloc[0, 1]
                impacts[feature] = float(corr)

            sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)

            print(f"   Results (Top 5 correlations):")
            for feat, corr in sorted_impacts[:5]:
                direction = "faster" if corr < 0 else "slower"
                priority_marker = " [PRIORITY - matches driver feedback]" if feat in priority_features else ""
                print(f"      • {feat:25s}: {corr:+.3f} ({direction} with increase){priority_marker}")

            analysis_results = {
                'method': 'correlation',
                'all_impacts': impacts,
                'most_impactful': sorted_impacts[0]
            }

        else:  # regression
            # Full regression model
            print(f"\n   [STATS] Running regression analysis...")
            model_df = df.dropna(subset=[target] + selected_features)

            y = model_df[target]
            X = model_df[selected_features]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LinearRegression()
            model.fit(X_scaled, y)

            impacts = {feat: float(coef) for feat, coef in zip(selected_features, model.coef_)}
            sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)

            r_squared = model.score(X_scaled, y)
            print(f"   Model R² = {r_squared:.3f}")
            print(f"   Results (Top 5 impacts):")
            for feat, impact in sorted_impacts[:5]:
                direction = "REDUCE" if impact > 0 else "INCREASE"
                print(f"      • {feat:25s}: {impact:+.3f} ({direction} to improve)")

            analysis_results = {
                'method': 'regression',
                'all_impacts': impacts,
                'most_impactful': sorted_impacts[0],
                'r_squared': r_squared
            }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

    # Prepare state update
    updated_state = {
        "analysis": analysis_results,
        "analysis_strategy": strategy,
        "selected_features": selected_features
    }

    # Merge with existing state for visibility
    merged_state = {**state, **updated_state}
    _print_state_transition("AGENT 2: Data Scientist", "AGENT 3: Crew Chief", merged_state)

    return updated_state

# ========== AGENT 3: CREW CHIEF (Recommendation Synthesis) ==========
def engineer_agent(state: RaceEngineerState):
    """
    Agent 3: Recommendation Strategist
    DECISIONS:
    - Single-parameter or multi-parameter recommendation?
    - Strong signal or weak signal?
    - What magnitude of change to suggest?
    """
    print("\n[AGENT 3] Crew Chief: Synthesizing recommendations...")

    analysis = state.get('analysis')
    strategy = state.get('analysis_strategy', 'unknown')
    driver_diagnosis = state.get('driver_diagnosis', {})
    priority_features = driver_diagnosis.get('priority_features', [])

    if not analysis:
        return {"error": "No analysis results"}

    param, impact = analysis['most_impactful']
    method = analysis.get('method', 'unknown')

    print(f"   [ANALYSIS] Analysis method used: {method.upper()}")
    print(f"   [TARGET] Top parameter from data: {param}")
    print(f"   [STATS] Impact magnitude: {abs(impact):.3f}")

    # DECISION 0: Should we trust data or driver feedback?
    all_impacts = analysis.get('all_impacts', {})
    recommended_param = param
    recommended_impact = impact
    decision_rationale = ""

    if driver_diagnosis and priority_features:
        if param in priority_features:
            print(f"   [VALIDATED] VALIDATION: Top parameter matches driver feedback!")
            print(f"      Driver complaint: {driver_diagnosis.get('diagnosis')}")
            print(f"      Data confirms: {param} is primary factor")
            print(f"    DECISION: Trust the data - driver intuition validated")
            decision_rationale = "driver_validated_by_data"
        else:
            # Data contradicts driver - need to make a decision
            print(f"   [WARNING]  CONFLICT: Data contradicts driver feedback")
            print(f"      Driver complaint: {driver_diagnosis.get('diagnosis')}")
            print(f"      Data top parameter: {param} (not in driver's priority list)")

            # Find strongest parameter from driver's priority list
            priority_impacts = {p: all_impacts.get(p, 0) for p in priority_features if p in all_impacts}

            if priority_impacts:
                best_priority_param = max(priority_impacts.items(), key=lambda x: abs(x[1]))
                best_priority_name, best_priority_impact = best_priority_param

                print(f"      Strongest priority parameter: {best_priority_name} ({best_priority_impact:+.3f})")
                print(f"\n    DECISION: Prioritize driver feedback")
                print(f"      Rationale: Driver has physical feel data we don't capture in telemetry.")
                print(f"      Action: Recommend {best_priority_name} (aligns with driver complaint)")
                print(f"      Note: Data suggests {param} but will test driver-relevant parameter first")

                recommended_param = best_priority_name
                recommended_impact = best_priority_impact
                decision_rationale = "driver_feedback_prioritized"
            else:
                print(f"\n    DECISION: Trust the data (no strong correlations in driver's priority areas)")
                decision_rationale = "data_prioritized_no_alternatives"
    else:
        print(f"    [INFO] No driver feedback - using pure data-driven recommendation")
        decision_rationale = "data_only"

    # Update the parameter we'll actually recommend
    param = recommended_param
    impact = recommended_impact

    # DECISION 0.5: Validate against driver constraints
    driver_constraints = state.get('driver_constraints', {})
    constraint_violated = False
    constraint_message = ""

    if driver_constraints:
        parameter_limits = driver_constraints.get('parameter_limits', [])
        already_tried = driver_constraints.get('already_tried', [])
        cannot_adjust = driver_constraints.get('cannot_adjust', [])

        # Check if recommended parameter is at a limit
        for limit in parameter_limits:
            if limit['param'] == param:
                # Determine if our recommendation violates this limit
                limit_type = limit['limit_type']
                recommending_increase = (method == "correlation" and impact < 0) or (method == "regression" and impact < 0)

                if (limit_type == "at_minimum" and not recommending_increase) or \
                   (limit_type == "at_maximum" and recommending_increase):
                    constraint_violated = True
                    constraint_message = f"\n    [CONSTRAINT] WARNING: {param} is {limit['limit_type']} - {limit['reason']}"
                    print(constraint_message)
                    print(f"    DECISION: Finding alternative parameter...")

                    # Find next best parameter that doesn't violate constraints
                    sorted_impacts = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
                    for alt_param, alt_impact in sorted_impacts:
                        # Skip if same as original
                        if alt_param == param:
                            continue

                        # Check if this alternative violates constraints
                        alt_violated = False
                        for alt_limit in parameter_limits:
                            if alt_limit['param'] == alt_param:
                                alt_recommending_increase = (method == "correlation" and alt_impact < 0) or (method == "regression" and alt_impact < 0)
                                if (alt_limit['limit_type'] == "at_minimum" and not alt_recommending_increase) or \
                                   (alt_limit['limit_type'] == "at_maximum" and alt_recommending_increase):
                                    alt_violated = True
                                    break

                        if not alt_violated and alt_param not in cannot_adjust:
                            print(f"    ALTERNATIVE: Recommending {alt_param} instead (impact: {abs(alt_impact):.3f})")
                            param = alt_param
                            impact = alt_impact
                            decision_rationale = "constraint_aware"
                            break

        # Check if parameter cannot be adjusted
        if param in cannot_adjust:
            constraint_violated = True
            constraint_message = f"\n    [CONSTRAINT] WARNING: {param} cannot be adjusted per driver"
            print(constraint_message)

        # Check if already tried
        if param in already_tried:
            print(f"    [INFO] NOTE: {param} was already tried - recommending anyway based on new data")

    # DECISION 1: Determine signal strength
    if abs(impact) > 0.1:
        signal_strength = "STRONG"
        print(f"    DECISION: Strong signal detected (|{impact:.3f}| > 0.1 threshold)")
    elif abs(impact) > 0.05:
        signal_strength = "MODERATE"
        print(f"    DECISION: Moderate signal (|{impact:.3f}| > 0.05)")
    else:
        signal_strength = "WEAK"
        print(f"    DECISION: Weak signal (|{impact:.3f}| < 0.05)")

    # DECISION 2: Generate appropriate recommendation with specific values
    if signal_strength == "STRONG":
        # Calculate specific adjustment recommendation
        setup_knowledge = state.get('setup_knowledge_base', {})
        param_limits = setup_knowledge.get('limits', {}).get(param, {})

        # Determine adjustment magnitude based on correlation strength
        if method == "correlation":
            increasing = impact < 0  # Negative correlation means increase to go faster

            # Calculate suggested adjustment
            adjustment_str = ""
            rationale_str = ""

            if 'tire_psi' in param:
                # Tire pressure: suggest 1-2 PSI changes
                if abs(impact) > 0.4:
                    adjustment = 2.0
                elif abs(impact) > 0.2:
                    adjustment = 1.5
                else:
                    adjustment = 1.0

                direction_word = "Increase" if increasing else "Reduce"
                adjustment_str = f"{direction_word} {param} by {adjustment} PSI"

                if increasing:
                    rationale_str = "Higher pressure increases tire stiffness and reduces contact patch, improving responsiveness"
                else:
                    rationale_str = "Lower pressure increases contact patch and mechanical grip, improving traction"

            elif 'spring' in param:
                # Springs: suggest 25-50 lb/in changes
                if abs(impact) > 0.4:
                    adjustment = 50
                elif abs(impact) > 0.2:
                    adjustment = 37.5
                else:
                    adjustment = 25

                direction_word = "Stiffen" if not increasing else "Soften"
                adjustment_str = f"{direction_word} {param} by {adjustment:.0f} lb/in"

                if not increasing:
                    rationale_str = "Stiffer springs reduce body roll and improve platform stability in corners"
                else:
                    rationale_str = "Softer springs improve mechanical grip by maintaining tire contact through bumps"

            elif 'cross_weight' in param:
                # Cross weight: suggest 0.5-1.0% changes
                if abs(impact) > 0.4:
                    adjustment = 1.0
                elif abs(impact) > 0.2:
                    adjustment = 0.75
                else:
                    adjustment = 0.5

                direction_word = "Increase" if not increasing else "Reduce"
                adjustment_str = f"{direction_word} {param} by {adjustment}%"

                if not increasing:
                    rationale_str = "More cross weight transfers load to right-rear, improving turn entry bite"
                else:
                    rationale_str = "Less cross weight reduces turn entry push and improves front grip"

            elif 'track_bar' in param:
                # Track bar: suggest 0.25-0.5 inch changes
                if abs(impact) > 0.4:
                    adjustment = 0.5
                elif abs(impact) > 0.2:
                    adjustment = 0.375
                else:
                    adjustment = 0.25

                direction_word = "Raise" if not increasing else "Lower"
                adjustment_str = f"{direction_word} {param} by {adjustment} inches"

                if not increasing:
                    rationale_str = "Raising track bar shifts rear roll center, tightening the car"
                else:
                    rationale_str = "Lowering track bar shifts rear roll center, loosening the car"
            else:
                # Generic parameter
                direction_word = "Increase" if increasing else "Reduce"
                adjustment_str = f"{direction_word} {param}"
                rationale_str = "Correlation analysis indicates this adjustment will improve lap times"

            # Add context based on how we made the decision
            context_note = ""
            if decision_rationale == "driver_validated_by_data":
                context_note = f"\n   Addresses driver complaint: {driver_diagnosis.get('diagnosis', '')}"
            elif decision_rationale == "driver_feedback_prioritized":
                context_note = f"\n   Prioritizes driver complaint: {driver_diagnosis.get('diagnosis', '')}"
                context_note += f"\n   Note: Data suggested different parameter, but trusting driver expertise"

            rec = f"STRONG RECOMMENDATION: {adjustment_str} (correlation: {impact:.3f})\n" + \
                  f"   Expected effect: {rationale_str}{context_note}"

        else:  # regression
            direction = "REDUCE" if impact > 0 else "INCREASE"
            rec = f"PRIMARY FOCUS: {direction} {param}\n" + \
                  f"   Predicted impact: {abs(impact):.3f}s per standardized unit\n" + \
                  f"   Confidence: High (regression coefficient)"

        print(f"    DECISION: Single-parameter recommendation ({param})")

    elif signal_strength == "MODERATE":
        direction = "increase" if (impact < 0 or method == "regression" and impact < 0) else "reduce"
        rec = f"MODERATE RECOMMENDATION: Consider {direction} {param}\n" + \
              f"   Impact: {abs(impact):.3f}\n" + \
              f"   Also monitor secondary parameters"

        print(f"    DECISION: Single-parameter with caveats")

    else:  # WEAK
        all_impacts = analysis.get('all_impacts', {})
        top_3 = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        params = [p for p, _ in top_3]

        rec = f"NO DOMINANT PARAMETER FOUND\n" + \
              f"   Top parameter '{param}' has only {abs(impact):.3f} impact\n" + \
              f"   Recommendation: Test interaction effects between:\n" + \
              f"      • " + "\n      • ".join(params)

        print(f"    DECISION: Multi-parameter interaction testing recommended")

    # DECISION 3: Add context-specific advice
    all_impacts = analysis.get('all_impacts', {})
    if all_impacts:
        sorted_all = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        if len(sorted_all) > 1:
            print(f"\n   [INFO]  Secondary factors to monitor:")
            for p, i in sorted_all[1:min(4, len(sorted_all))]:
                print(f"      • {p:25s}: {i:+.3f}")

    # DECISION 4 (OPTIONAL): Generate LLM explanation with knowledge base context
    # This showcases API integration and prompt engineering skills
    try:
        from llm_explainer import generate_llm_explanation

        decision_context = {
            'driver_complaint': driver_diagnosis.get('diagnosis', 'None'),
            'data_top_param': analysis['most_impactful'][0],
            'data_correlation': analysis['most_impactful'][1],
            'priority_features': priority_features,
            'decision_type': decision_rationale,
            'recommended_param': param,
            'recommended_impact': impact
        }

        # Add knowledge base context if available
        setup_knowledge = state.get('setup_knowledge_base')
        if setup_knowledge:
            from knowledge_base_loader import get_relevant_knowledge, format_knowledge_for_llm

            complaint_type = driver_diagnosis.get('complaint_type', 'general')
            relevant_knowledge = get_relevant_knowledge(
                setup_knowledge,
                handling_issue=complaint_type,
                parameter=param
            )

            if relevant_knowledge:
                knowledge_context = format_knowledge_for_llm(relevant_knowledge)
                decision_context['knowledge_base'] = knowledge_context

                print(f"\n   [KNOWLEDGE BASE] Consulting setup manual...")
                # Show parameter limits if available
                if relevant_knowledge.get('limits'):
                    for p, limits in relevant_knowledge['limits'].items():
                        print(f"      {p}: {limits['min']}-{limits['max']} {limits['unit']}")

        llm_explanation = generate_llm_explanation(decision_context)
        if llm_explanation:
            print(f"\n   [CREW CHIEF PERSPECTIVE]")
            print(f"   {llm_explanation}")

        # DECISION 4.5: Generate driver acknowledgment for non-technical aspects
        # This addresses tone, personality, or comments not captured in technical diagnosis
        raw_driver_feedback = state.get('raw_driver_feedback')
        if raw_driver_feedback:
            from llm_explainer import generate_driver_acknowledgment

            driver_ack = generate_driver_acknowledgment(
                raw_driver_feedback,
                driver_diagnosis.get('diagnosis', 'General handling issue')
            )

            if driver_ack:
                print(f"\n   [DRIVER ACKNOWLEDGMENT]")
                print(f"   {driver_ack}")

    except ImportError:
        # llm_explainer module not available, skip this step
        pass
    except Exception as e:
        # Don't fail the whole workflow if LLM explanation fails
        print(f"\n   [INFO] LLM explanation unavailable: {str(e)[:50]}")

    # DECISION 5 (OPTIONAL): Multi-Turn Session Analysis
    # Analyze patterns across multiple stints for iterative learning
    session_history = state.get('session_history', [])
    if session_history and len(session_history) >= 1:
        try:
            from llm_explainer import generate_llm_multi_turn_analysis

            print(f"\n   [SESSION LEARNING] Analyzing patterns from {len(session_history)} previous stint(s)...")

            current_context = {
                'driver_complaint': driver_diagnosis.get('diagnosis', 'None'),
                'recommended_param': param,
                'recommended_impact': impact,
                'recommendation': rec
            }

            insights = generate_llm_multi_turn_analysis(
                session_history,
                current_context
            )

            if insights:
                print(f"\n   [MULTI-STINT INSIGHTS]")
                print(f"   {insights}")

        except ImportError:
            pass
        except Exception as e:
            print(f"\n   [INFO] Multi-turn analysis unavailable: {str(e)[:50]}")

    return {"recommendation": rec}

# == ERROR HANDLER ==
def error_handler(state: RaceEngineerState):
    """Handle errors gracefully"""
    error = state.get('error', 'Unknown error occurred')
    print(f"\n[ERROR] {error}")
    return state

# == GRAPH CONSTRUCTION ==
def create_race_engineer_workflow():
    """
    Build LangGraph workflow with 3 decision-making agents
    Each agent makes visible, strategic choices
    """

    # Initialize the graph
    workflow = StateGraph(RaceEngineerState)

    # Add agent nodes
    workflow.add_node("telemetry", telemetry_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("engineer", engineer_agent)
    workflow.add_node("error", error_handler)

    # Conditional routing based on agent decisions
    def check_telemetry_output(state):
        """Route based on data quality assessment"""
        if state.get('error'):
            return "error"
        return "analysis"

    def check_analysis_output(state):
        """Route based on analysis success"""
        if state.get('error'):
            return "error"
        return "engineer"

    # Set entry point
    workflow.set_entry_point("telemetry")

    # Add conditional edges (decision-based routing)
    workflow.add_conditional_edges("telemetry", check_telemetry_output)
    workflow.add_conditional_edges("analysis", check_analysis_output)

    # Terminal nodes
    workflow.add_edge("engineer", END)
    workflow.add_edge("error", END)

    # Compile the graph
    app = workflow.compile()

    return app

# Create the application instance
app = create_race_engineer_workflow()


# ========== MODULARITY DEMONSTRATION ==========
# These can be swapped out for different analysis approaches

class AnalysisStrategy:
    """Base class - demonstrates modularity"""
    def analyze(self, df, features):
        raise NotImplementedError

class RegressionStrategy(AnalysisStrategy):
    """Swappable strategy: Linear regression"""
    def analyze(self, df, features):
        # Implementation would go here
        pass

class CorrelationStrategy(AnalysisStrategy):
    """Swappable strategy: Simple correlation"""
    def analyze(self, df, features):
        # Implementation would go here
        pass

# Could easily add: RandomForestStrategy, NeuralNetStrategy, etc.

# == This is how you run the entire system ==
if __name__ == "__main__":
    print("🏁 Bristol AI Race Engineer - Initializing...")
    
    # --- THIS IS THE KEY CHANGE ---
    # Instead of mock files, we tell it to find ALL .ldx files
    # in a 'bristol_data' folder.
    
    # 1. Go create a folder named 'bristol_data' inside your
    #    'AI Race Engineer' project folder.
    # 2. Drag all 30+ of your .ldx files from your data
    #    collection sprint into this 'bristol_data' folder.
    
    data_directory = Path("bristol_data")
    
    # This line finds every file ending in .ldx in that folder
    all_ldx_files = list(data_directory.glob("*.ldx"))

    if not all_ldx_files:
        print("="*30)
        print("[ERROR] ERROR: No .ldx files found in 'bristol_data' folder.")
        print("   Please create 'bristol_data' and add your files.")
        print("="*30)
    else:
        # 2. This is the input for our graph
        inputs = { "ldx_file_paths": all_ldx_files }
        
        # 3. Run the graph!
        final_state = app.invoke(inputs)
        
        print("\n" + "="*30)
        print("[VALIDATED] Run Complete. Final Recommendation:")
        print(final_state.get('recommendation') or final_state.get('error', 'Unknown error'))
        print("="*30)