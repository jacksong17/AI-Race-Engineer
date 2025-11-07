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
    data_quality_decision: Optional[str]
    analysis_strategy: Optional[str]
    selected_features: Optional[List[str]]
    analysis: Optional[Dict]
    recommendation: Optional[str]
    error: Optional[str]

# ========== AGENT 1: TELEMETRY CHIEF (Data Quality + Parsing) ==========
def telemetry_agent(state: RaceEngineerState):
    """
    Agent 1: Data Quality Assessor + Parser
    DECISIONS:
    - Remove outliers or keep them?
    - Sufficient data for analysis?
    - Which sessions are valid?
    """
    print("\n[AGENT 1] Telemetry Chief: Assessing data quality...")

    df = state.get('raw_setup_data')
    if df is None or df.empty:
        return {"error": "No data provided"}

    # DECISION 1: Outlier detection
    lap_times = df['fastest_time']
    print(f"   📊 Dataset: {len(df)} sessions")
    print(f"   📈 Lap time range: {lap_times.min():.3f}s - {lap_times.max():.3f}s")
    print(f"   📉 Variance: {lap_times.std():.3f}s")

    # Calculate IQR for outlier detection
    q1, q3 = lap_times.quantile(0.25), lap_times.quantile(0.75)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    outliers = df[lap_times > outlier_threshold]

    # Agent makes decision about outliers
    if len(outliers) > 0:
        print(f"   ⚠️  Found {len(outliers)} outlier(s) > {outlier_threshold:.3f}s")
        if len(outliers) < len(df) * 0.2:  # Less than 20%
            df_clean = df[lap_times <= outlier_threshold]
            print(f"   ✓ DECISION: Removing {len(outliers)} outliers (keeping {len(df_clean)} sessions)")
            decision = f"removed_{len(outliers)}_outliers"
        else:
            print(f"   ✓ DECISION: Keeping all data (outliers represent >20% of dataset)")
            df_clean = df
            decision = "kept_all_data"
    else:
        print(f"   ✓ DECISION: No outliers detected, proceeding with all {len(df)} sessions")
        df_clean = df
        decision = "no_outliers_found"

    # DECISION 2: Sample size check
    if len(df_clean) < 5:
        return {"error": f"Insufficient data: only {len(df_clean)} valid sessions (need 5+)"}

    print(f"   ✓ Data quality check passed: {len(df_clean)} sessions ready for analysis")

    return {
        "raw_setup_data": df_clean,
        "data_quality_decision": decision
    }

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

    # DECISION 1: Feature selection (dynamic based on variance)
    potential_features = [
        'tire_psi_lf', 'tire_psi_rf', 'tire_psi_lr', 'tire_psi_rr',
        'cross_weight', 'track_bar_height_left', 'spring_lf', 'spring_rf'
    ]

    selected_features = []
    print(f"   🔍 Evaluating {len(potential_features)} potential features...")

    for feature in potential_features:
        if feature not in df.columns:
            continue

        variance = df[feature].std()
        # Agent decides: only use features that actually varied during testing
        if variance > 0.01:  # Feature was changed meaningfully
            selected_features.append(feature)
            print(f"      ✓ {feature:25s} (varied: σ={variance:.2f})")
        else:
            print(f"      ✗ {feature:25s} (constant: σ={variance:.4f})")

    if len(selected_features) < 2:
        return {"error": f"Insufficient variable features: only {len(selected_features)} found"}

    print(f"   ✓ DECISION: Using {len(selected_features)} features for analysis")

    # DECISION 2: Choose analysis strategy
    sample_size = len(df)
    feature_count = len(selected_features)
    variance = df['fastest_time'].std()

    print(f"\n   📊 Evaluating strategy options...")
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

    print(f"   ✓ DECISION: Using {strategy.upper()} analysis ({reason})")

    # Execute chosen strategy
    target = 'fastest_time'

    try:
        if strategy == "correlation":
            # Simple correlation analysis
            impacts = {}
            print(f"\n   📈 Running correlation analysis...")
            for feature in selected_features:
                corr = df[[feature, target]].corr().iloc[0, 1]
                impacts[feature] = float(corr)

            sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)

            print(f"   Results (Top 5 correlations):")
            for feat, corr in sorted_impacts[:5]:
                direction = "faster" if corr < 0 else "slower"
                print(f"      • {feat:25s}: {corr:+.3f} ({direction} with increase)")

            analysis_results = {
                'method': 'correlation',
                'all_impacts': impacts,
                'most_impactful': sorted_impacts[0]
            }

        else:  # regression
            # Full regression model
            print(f"\n   📈 Running regression analysis...")
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

    return {
        "analysis": analysis_results,
        "analysis_strategy": strategy,
        "selected_features": selected_features
    }

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

    if not analysis:
        return {"error": "No analysis results"}

    param, impact = analysis['most_impactful']
    method = analysis.get('method', 'unknown')

    print(f"   📊 Analysis method used: {method.upper()}")
    print(f"   🎯 Top parameter: {param}")
    print(f"   📈 Impact magnitude: {abs(impact):.3f}")

    # DECISION 1: Determine signal strength
    if abs(impact) > 0.1:
        signal_strength = "STRONG"
        print(f"   ✓ DECISION: Strong signal detected (|{impact:.3f}| > 0.1 threshold)")
    elif abs(impact) > 0.05:
        signal_strength = "MODERATE"
        print(f"   ✓ DECISION: Moderate signal (|{impact:.3f}| > 0.05)")
    else:
        signal_strength = "WEAK"
        print(f"   ✓ DECISION: Weak signal (|{impact:.3f}| < 0.05)")

    # DECISION 2: Generate appropriate recommendation
    if signal_strength == "STRONG":
        if method == "correlation":
            if impact < -0.1:
                rec = f"STRONG RECOMMENDATION: Increase {param} (strong negative correlation: {impact:.3f})\n" + \
                      f"   Expected effect: Significantly faster lap times"
            else:
                rec = f"STRONG RECOMMENDATION: Reduce {param} (strong positive correlation: {impact:.3f})\n" + \
                      f"   Expected effect: Significantly faster lap times"
        else:  # regression
            direction = "REDUCE" if impact > 0 else "INCREASE"
            rec = f"PRIMARY FOCUS: {direction} {param}\n" + \
                  f"   Predicted impact: {abs(impact):.3f}s per standardized unit\n" + \
                  f"   Confidence: High (regression coefficient)"

        print(f"   ✓ DECISION: Single-parameter recommendation")

    elif signal_strength == "MODERATE":
        direction = "increase" if (impact < 0 or method == "regression" and impact < 0) else "reduce"
        rec = f"MODERATE RECOMMENDATION: Consider {direction} {param}\n" + \
              f"   Impact: {abs(impact):.3f}\n" + \
              f"   Also monitor secondary parameters"

        print(f"   ✓ DECISION: Single-parameter with caveats")

    else:  # WEAK
        all_impacts = analysis.get('all_impacts', {})
        top_3 = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        params = [p for p, _ in top_3]

        rec = f"NO DOMINANT PARAMETER FOUND\n" + \
              f"   Top parameter '{param}' has only {abs(impact):.3f} impact\n" + \
              f"   Recommendation: Test interaction effects between:\n" + \
              f"      • " + "\n      • ".join(params)

        print(f"   ✓ DECISION: Multi-parameter interaction testing recommended")

    # DECISION 3: Add context-specific advice
    all_impacts = analysis.get('all_impacts', {})
    if all_impacts:
        sorted_all = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        if len(sorted_all) > 1:
            print(f"\n   ℹ️  Secondary factors to monitor:")
            for p, i in sorted_all[1:min(4, len(sorted_all))]:
                print(f"      • {p:25s}: {i:+.3f}")

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
        print("✅ Run Complete. Final Recommendation:")
        print(final_state.get('recommendation') or final_state.get('error', 'Unknown error'))
        print("="*30)