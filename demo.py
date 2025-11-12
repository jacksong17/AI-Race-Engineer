"""
Bristol AI Race Engineer Demo
Unified interface with intelligent routing and concise output
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
import numpy as np
import io
from contextlib import redirect_stdout
from csv_data_loader import CSVDataLoader
from input_router import InputRouter, AnalysisRequest
from dotenv import load_dotenv

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    # Force UTF-8 encoding on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        # Fallback for older Python versions
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Load environment variables from .env file
load_dotenv()


def generate_mock_data():
    """Generate mock Bristol testing data for demo purposes"""
    # Bristol baseline setup (realistic values)
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

        # Vary parameters systematically
        if i < 5:
            # Baseline runs
            session['fastest_time'] = 15.543 + np.random.normal(0, 0.05)
        elif i < 8:
            # LF pressure tests - lower is better
            session['tire_psi_lf'] = 28.0 + (i - 5) * 2
            session['fastest_time'] = 15.543 - 0.05 * (8 - i) + np.random.normal(0, 0.03)
        elif i < 11:
            # Cross weight tests - higher is better
            session['cross_weight'] = 52.0 + (i - 8) * 2
            session['fastest_time'] = 15.543 - 0.08 * (i - 8) + np.random.normal(0, 0.03)
        elif i < 14:
            # Track bar tests
            session['track_bar_height_left'] = 5.0 + (i - 11) * 5
            session['fastest_time'] = 15.543 - 0.04 * (i - 11) + np.random.normal(0, 0.03)
        else:
            # Combined optimal settings
            session['tire_psi_lf'] = 26.0
            session['cross_weight'] = 56.0
            session['track_bar_height_left'] = 12.0
            session['fastest_time'] = 15.543 - 0.30 + np.random.normal(0, 0.02)

        sessions.append(session)

    return pd.DataFrame(sessions)


def load_data_silent():
    """Load data without verbose output"""
    loader = CSVDataLoader()
    df = loader.load_data()

    if df is not None:
        df = loader.prepare_for_ai_analysis(df)
        return df, True
    else:
        return generate_mock_data(), False


def run_analysis_silent(df: pd.DataFrame, driver_feedback_dict: dict, verbose: bool = True) -> dict:
    """Run AI analysis workflow with optional verbosity control

    Args:
        df: DataFrame with telemetry data
        driver_feedback_dict: Driver feedback information
        verbose: If True (default), show all agent activity. If False, suppress output.
    """
    from race_engineer import app
    from race_engineer.state import create_initial_state

    # Convert driver feedback dict to string for create_initial_state
    driver_feedback_str = driver_feedback_dict.get('description',
                          f"{driver_feedback_dict.get('complaint', 'general')} issue")

    # Create proper initial state using the state factory
    initial_state = create_initial_state(
        driver_feedback=driver_feedback_str,
        telemetry_files=[],  # Files already loaded in df
        constraints=None,
        config={
            "track": "bristol",
            "car_class": "nascar_truck",
            "conditions": "dry",
            "analysis_mode": "standard"
        }
    )

    # Add the pre-loaded data to state in the format expected by tools
    # Convert DataFrame to dict format matching load_telemetry output
    initial_state['telemetry_data'] = {
        "data": df.to_dict(orient='records'),
        "data_columns": list(df.columns),
        "num_sessions": len(df),
        "parameters": [col for col in df.columns if col not in ['fastest_time', 'track', 'session_id']],
        "source_format": "csv_data",
        "load_warnings": []
    }

    # Only suppress output if explicitly requested (verbose=False)
    if verbose:
        # Show all agent activity in real-time
        state = app.invoke(initial_state)
    else:
        # Capture and suppress output
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            state = app.invoke(initial_state)

    return state


def format_output(state: dict, df: pd.DataFrame, request: AnalysisRequest,
                  using_real_data: bool, verbose: bool = False) -> str:
    """Comprehensive output with complete findings and actionable recommendations"""

    output_lines = []

    # Header
    output_lines.append("\n" + "=" * 70)
    output_lines.append("AI RACE ENGINEER - FINAL ANALYSIS REPORT")
    output_lines.append("=" * 70)
    output_lines.append("")

    # Driver feedback summary
    if request.driver_feedback:
        fb = request.driver_feedback
        output_lines.append("DRIVER FEEDBACK:")
        output_lines.append(f"   Issue: {fb.complaint.replace('_', ' ').title()}")
        output_lines.append(f"   Description: {fb.description}")
        output_lines.append(f"   Severity: {fb.severity.title() if hasattr(fb, 'severity') else 'Not specified'}")
        output_lines.append("")

    # Data Analysis Findings
    output_lines.append("DATA ANALYSIS FINDINGS:")
    analysis = state.get('statistical_analysis', {})
    if analysis:
        method = analysis.get('method', 'correlation')
        output_lines.append(f"   Analysis Method: {method.title()}")

        all_impacts = analysis.get('correlations') or analysis.get('coefficients', {})
        if all_impacts:
            sorted_impacts = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
            output_lines.append(f"   Parameters Analyzed: {len(all_impacts)}")
            output_lines.append(f"   Sessions Analyzed: {len(df)}")
            output_lines.append("")
            output_lines.append("   Top 5 Impactful Parameters:")
            for i, (param, impact) in enumerate(sorted_impacts[:5], 1):
                direction = "Reduce" if impact > 0 else "Increase"
                output_lines.append(f"      {i}. {param:25s}  {direction:8s}  (correlation: {impact:+.3f})")
    else:
        output_lines.append("   No statistical analysis available")
    output_lines.append("")

    # Knowledge Expert Insights
    knowledge_insights = state.get('knowledge_insights', {})
    if knowledge_insights:
        output_lines.append("KNOWLEDGE EXPERT INSIGHTS:")
        param_guidance = knowledge_insights.get('parameter_guidance', {})
        if param_guidance:
            output_lines.append(f"   NASCAR Manual Guidance: {len(param_guidance)} parameters")
            for param, guidance in list(param_guidance.items())[:3]:
                output_lines.append(f"      • {param}: {guidance}")
        output_lines.append("")

    # Primary Recommendation
    output_lines.append("=" * 70)
    output_lines.append("PRIMARY SETUP RECOMMENDATION")
    output_lines.append("=" * 70)

    if state.get('final_recommendation'):
        final_rec = state['final_recommendation']
        primary = final_rec.get('primary')

        if primary:
            param_display = primary['parameter'].replace('_', ' ').title()
            output_lines.append(f"   Parameter:  {param_display}")
            output_lines.append(f"   Action:     {primary['direction'].title()} by {primary['magnitude']} {primary.get('magnitude_unit', 'units')}")
            output_lines.append(f"   Confidence: {int(primary.get('confidence', 0.8) * 100)}%")
            output_lines.append("")

            # Rationale
            rationale = primary.get('rationale', 'Based on statistical correlation with lap time')
            output_lines.append(f"   Rationale:")
            output_lines.append(f"      {rationale}")
            output_lines.append("")

            # Tool validations
            if primary.get('tool_validations'):
                validations = primary['tool_validations']
                output_lines.append("   Validation Results:")

                # Handle both dict and list formats
                if isinstance(validations, dict):
                    for tool_name, result in validations.items():
                        if isinstance(result, dict):
                            if result.get('is_valid'):
                                output_lines.append(f"      ✓ {tool_name}: PASSED")
                            elif 'error' in result:
                                output_lines.append(f"      ✗ {tool_name}: {result['error']}")
                elif isinstance(validations, list):
                    for item in validations:
                        if isinstance(item, dict):
                            for tool_name, result in item.items():
                                if isinstance(result, dict):
                                    if result.get('is_valid'):
                                        output_lines.append(f"      ✓ {tool_name}: PASSED")
                                    elif 'error' in result:
                                        output_lines.append(f"      ✗ {tool_name}: {result['error']}")
                output_lines.append("")
        else:
            output_lines.append("   No specific recommendation available")
            output_lines.append("")
    else:
        output_lines.append("   WARNING: No recommendation generated")
        output_lines.append("")

    # Secondary Recommendations
    if state.get('final_recommendation') and state['final_recommendation'].get('recommendations'):
        recs = state['final_recommendation']['recommendations']
        if len(recs) > 0:
            output_lines.append("SECONDARY RECOMMENDATIONS:")
            for i, rec in enumerate(recs[:3], 1):
                if isinstance(rec, dict):
                    param = rec.get('parameter', 'Unknown').replace('_', ' ').title()
                    direction = rec.get('direction', '').title()
                    output_lines.append(f"   {i}. {param}: {direction}")
            output_lines.append("")

    # Performance Summary
    best_time = float(df['fastest_time'].min())
    baseline_time = float(df['fastest_time'].max())
    improvement = baseline_time - best_time

    output_lines.append("=" * 70)
    output_lines.append("PERFORMANCE SUMMARY")
    output_lines.append("=" * 70)
    output_lines.append(f"   Current Best Lap:    {best_time:.3f}s")
    output_lines.append(f"   Current Worst Lap:   {baseline_time:.3f}s")
    output_lines.append(f"   Current Spread:      {improvement:.3f}s")
    output_lines.append(f"   Estimated Potential: {best_time - 0.050:.3f}s  (improvement: 0.050s)")
    output_lines.append("")

    # Next Steps
    output_lines.append("=" * 70)
    output_lines.append("NEXT STEPS")
    output_lines.append("=" * 70)
    output_lines.append("   1. Implement the primary recommendation in your setup")
    output_lines.append("   2. Run 3-5 test laps to evaluate the change")
    output_lines.append("   3. Collect driver feedback on handling improvement")
    output_lines.append("   4. If positive, consider secondary recommendations")
    output_lines.append("   5. Document changes and lap times for future reference")
    output_lines.append("")

    # Data source
    data_source = "Real telemetry" if using_real_data else "Mock demo data"
    track_name = df['track'].iloc[0] if 'track' in df.columns and len(df) > 0 else 'bristol'
    output_lines.append(f"Data Source: {data_source} ({len(df)} sessions from {track_name})")
    output_lines.append("")
    output_lines.append("=" * 70)

    return "\n".join(output_lines)




# ===== MAIN EXECUTION =====

def run_demo(user_input: str = None, verbose: bool = False):
    """
    Run demo with unified interface

    Args:
        user_input: Optional user input with driver feedback
        verbose: Show detailed output
    """

    # Default example if no input provided
    if user_input is None:
        user_input = "The car feels really loose coming off corners in turns 1 and 2. The rear end wants to come around when I get on the throttle."

    # Parse input using intelligent router
    router = InputRouter()
    request = router.parse_input(user_input)

    # Override verbosity if requested
    if verbose:
        request.verbosity = 'detailed'

    # Load data silently
    df, using_real_data = load_data_silent()

    # Convert driver feedback to dict format
    if request.driver_feedback:
        driver_feedback_dict = router.create_driver_feedback_dict(request.driver_feedback)
    else:
        # Use neutral default if no driver feedback detected
        driver_feedback_dict = {
            'complaint': 'general_handling',
            'description': 'General setup optimization',
            'severity': 'minor',
            'phase': 'general'
        }

    # Show processing indicator
    print("\nAnalyzing setup data and driver feedback...\n")

    # Run analysis with full observability (always verbose by default)
    # User can see all agent activity, tool calls, and reasoning steps
    state = run_analysis_silent(df, driver_feedback_dict, verbose=True)

    # Check for errors
    if 'error' in state and state['error']:
        print(f"❌ Error: {state['error']}")
        sys.exit(1)

    # Format and display output
    output = format_output(state, df, request, using_real_data,
                          verbose=(request.verbosity == 'detailed'))
    print(output)

    # Save results
    results = {
        'user_input': user_input,
        'analysis_type': request.analysis_type,
        'focus_areas': request.focus_areas,
        'data_source': 'real_csv_data' if using_real_data else 'mock_data',
        'recommendation': state.get('recommendation', 'No recommendation'),
        'analysis': state.get('analysis', {}),
        'best_time': float(df['fastest_time'].min()),
        'baseline_time': float(df['fastest_time'].max()),
        'improvement': float(df['fastest_time'].max()) - float(df['fastest_time'].min()),
        'num_sessions': len(df)
    }

    output_path = Path("output/demo_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    # Check for command line arguments
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    # Remove flags from argv
    args = [arg for arg in sys.argv[1:] if arg not in ['--verbose', '-v']]

    if args:
        # User provided input as command line argument
        user_input = ' '.join(args)
        run_demo(user_input, verbose)
    else:
        # Run with default example
        run_demo(verbose=verbose)
