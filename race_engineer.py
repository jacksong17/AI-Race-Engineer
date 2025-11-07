# race_engineer.py
import pandas as pd
from typing import TypedDict, List, Dict, Optional
from pathlib import Path
from langgraph.graph import StateGraph, END

# --- NEW IMPORTS ---
# Import your own parser
from telemetry_parser import TelemetryParser 
# Import the tools for real analysis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# == This is your project's "memory" ==
# (RaceEngineerState stays the same)
class RaceEngineerState(TypedDict):
    ldx_file_paths: List[Path]
    raw_setup_data: Optional[pd.DataFrame]
    lap_statistics: Optional[pd.DataFrame]
    analysis: Optional[Dict]
    recommendation: Optional[str]
    error: Optional[str]

# == AGENT 1: UPGRADED Telemetry Chief ==
def telemetry_agent(state: RaceEngineerState):
    """
    Agent 1: Parses all .ldx files and combines them.
    """
    print("[TELEMETRY] Telemetry Chief: Processing setup files...")
    
    # 1. Check if we even have files
    file_paths = state.get('ldx_file_paths', [])
    if not file_paths:
        return {"error": "TelemetryAgent: No .ldx files found."}
    
    print(f"   > Found {len(file_paths)} files to process.")
    
    # 2. Instantiate your parser
    parser = TelemetryParser()
    all_data = []

    # 3. Loop through every file and parse it
    for file_path in file_paths:
        try:
            # This calls your .ldx parser!
            session_data = parser.parse_ldx_file(file_path)
            all_data.append(session_data)
        except Exception as e:
            print(f"   > ⚠️ Warning: Failed to parse {file_path.name}: {e}")

    if not all_data:
        return {"error": "TelemetryAgent: No data successfully processed."}
    
    # 4. Convert to DataFrame and add features
    df = pd.DataFrame(all_data)
    # This re-uses the logic from your parser to add rake, etc.
    df = parser._add_derived_features(df) 
    
    print(f"   > Successfully created DataFrame with {len(df)} runs.")
    
    # 5. Pass the real data to the next agent
    return {"raw_setup_data": df}

# == AGENT 2: UPGRADED Data Scientist ==
def analysis_agent(state: RaceEngineerState):
    """
    Agent 2: Runs regression to find correlations.
    """
    print("[ANALYSIS] Data Scientist: Analyzing performance data...")
    df = state.get('raw_setup_data')
    
    if df is None or df.empty:
        return {"error": "AnalysisAgent: No data to analyze."}

    # --- This is your new "brain" ---
    
    # 1. Define what we want to test
    # We'll use your best lap time as the target
    target = 'fastest_time'
    # These are the setup parameters we'll test
    features = [
        'tire_psi_lf', 'tire_psi_rf', 'tire_psi_lr', 'tire_psi_rr',
        'cross_weight', 'track_bar_height_left', 'spring_lf', 'spring_rf'
    ]
    
    # 2. Clean the data for modeling
    # Drop any runs where we're missing this data
    model_df = df.dropna(subset=[target] + features)
    
    if len(model_df) < 5: # Need enough data to model
        return {"error": f"AnalysisAgent: Not enough data for modeling (found {len(model_df)} valid runs)."}

    print(f"   > Running regression on {len(model_df)} valid runs.")

    # 3. Prepare data for the model
    y = model_df[target]
    X = model_df[features]
    
    # Scale data (e.g., 10 PSI and 50% cross weight) so they
    # can be compared fairly.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Run the Linear Regression
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # 5. Get the results
    # These are the "coefficients" or "impact scores"
    coefficients = model.coef_
    
    # Create a simple dictionary of the results
    param_impact = {feature: coeff for feature, coeff in zip(features, coefficients)}
    
    # Sort by impact
    sorted_impact = sorted(param_impact.items(), key=lambda item: abs(item[1]), reverse=True)
    
    print("   > Model Results (Impact on Lap Time):")
    for param, impact in sorted_impact:
        print(f"     - {param}: {impact:.3f}")
        
    analysis_results = {
        "most_impactful": sorted_impact[0],
        "all_impacts": param_impact
    }
    
    return {"analysis": analysis_results}

# == AGENT 3: UPGRADED Crew Chief ==
def engineer_agent(state: RaceEngineerState):
    """
    Agent 3: Translates stats into a setup recommendation.
    """
    print("[ENGINEER] Crew Chief: Generating recommendation...")
    analysis = state.get('analysis')
    
    if analysis is None:
        return {"error": "EngineerAgent: No analysis to review."}

    # Get the top finding from the model
    param, impact = analysis['most_impactful']
    
    # A negative impact is GOOD (it reduces lap time)
    if impact < -0.1: # -0.1 is our threshold for a "strong" finding
        recommendation_text = f"Key Finding: **Increase** '{param}'. It has a strong negative impact ({impact:.3f}) on lap time. (Higher value = lower time)"
    elif impact > 0.1:
        recommendation_text = f"Key Finding: **Reduce** '{param}'. It has a strong positive impact ({impact:.3f}) on lap time. (Lower value = lower time)"
    else:
        recommendation_text = f"No strong single-parameter impact found. '{param}' was closest ({impact:.3f}). Hold setup and test interaction effects."

    print(f"   > {recommendation_text}")
    return {"recommendation": recommendation_text}

# == ERROR HANDLER ==
def error_handler(state: RaceEngineerState):
    """Handle errors gracefully"""
    error = state.get('error', 'Unknown error occurred')
    print(f"[ERROR] Error: {error}")
    return state

# == GRAPH CONSTRUCTION ==
def create_race_engineer_workflow():
    """Build and compile the LangGraph workflow"""

    # Initialize the graph
    workflow = StateGraph(RaceEngineerState)

    # Add nodes
    workflow.add_node("telemetry", telemetry_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("engineer", engineer_agent)
    workflow.add_node("error", error_handler)

    # Define conditional routing
    def check_telemetry_output(state):
        """Route to analysis or error based on telemetry agent output"""
        if state.get('error'):
            return "error"
        return "analysis"

    def check_analysis_output(state):
        """Route to engineer or error based on analysis agent output"""
        if state.get('error'):
            return "error"
        return "engineer"

    # Set entry point
    workflow.set_entry_point("telemetry")

    # Add edges with conditional routing
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