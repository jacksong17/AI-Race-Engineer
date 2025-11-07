"""
Bristol AI Race Engineer Demo
Unified interface with intelligent routing and concise output
"""

import sys
from pathlib import Path
import pandas as pd
import json
import numpy as np
import io
from contextlib import redirect_stdout
from csv_data_loader import CSVDataLoader
from input_router import InputRouter, AnalysisRequest


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


def run_analysis_silent(df: pd.DataFrame, driver_feedback_dict: dict) -> dict:
    """Run AI analysis workflow silently in background"""
    from race_engineer import app

    initial_state = {
        'raw_setup_data': df,
        'driver_feedback': driver_feedback_dict,
        'data_quality_decision': None,
        'analysis_strategy': None,
        'selected_features': None,
        'analysis': None,
        'recommendation': None,
        'error': None
    }

    # Capture all verbose output from agents
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        state = app.invoke(initial_state)

    return state


def format_output(state: dict, df: pd.DataFrame, request: AnalysisRequest,
                  using_real_data: bool, verbose: bool = False) -> str:
    """Format output - concise by default, detailed if requested"""

    output_lines = []

    # Header
    output_lines.append("=" * 60)
    output_lines.append("AI RACE ENGINEER")
    output_lines.append("=" * 60)
    output_lines.append("")

    # Driver feedback summary (if present)
    if request.driver_feedback:
        fb = request.driver_feedback
        output_lines.append(f"🎧 Driver Feedback: {fb.complaint.replace('_', ' ').title()}")
        output_lines.append(f"   {fb.description}")
        output_lines.append("")

    # Primary recommendation
    recommendation = state.get('recommendation', 'No recommendation available')
    output_lines.append("💡 RECOMMENDATION:")
    output_lines.append(f"   {recommendation}")
    output_lines.append("")

    # Top 3 impacts (concise) or Top 5 (verbose)
    analysis = state.get('analysis', {})
    if analysis:
        all_impacts = analysis.get('all_impacts', {})
        if all_impacts:
            sorted_impacts = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
            num_to_show = 5 if verbose else 3

            output_lines.append("📊 KEY PARAMETERS:")
            for param, impact in sorted_impacts[:num_to_show]:
                direction = "↓" if impact > 0 else "↑"
                action = "Reduce" if impact > 0 else "Increase"
                output_lines.append(f"   {direction} {param:20s}  {action:8s}  ({impact:+.3f}s)")
            output_lines.append("")

    # Performance summary
    best_time = float(df['fastest_time'].min())
    baseline_time = float(df['fastest_time'].max())
    improvement = baseline_time - best_time

    output_lines.append("⚡ PERFORMANCE:")
    output_lines.append(f"   Current Best:  {best_time:.3f}s")
    output_lines.append(f"   Potential:     {baseline_time - improvement - 0.050:.3f}s  (↓{improvement + 0.050:.3f}s)")
    output_lines.append("")

    # Additional info if verbose
    if verbose:
        data_source = "Real telemetry" if using_real_data else "Mock demo data"
        output_lines.append(f"📁 Data: {data_source} ({len(df)} sessions)")
        output_lines.append("")

    output_lines.append("=" * 60)

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
    print("\n🔧 Analyzing setup data and driver feedback...\n")

    # Run analysis silently in background
    state = run_analysis_silent(df, driver_feedback_dict)

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
