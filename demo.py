"""
Simplified Bristol AI Race Engineer Demo
Runs the core AI analysis without visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import json
import numpy as np
from datetime import datetime
from csv_data_loader import CSVDataLoader
from session_manager import SessionManager


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


# ===== MAIN EXECUTION =====

print("="*70)
print("  BRISTOL AI RACE ENGINEER - SIMPLIFIED DEMO")
print("="*70)
print()

# Step 1: Load real data or generate mock data
print("[1/5] Loading training data...")
print()

# Try to load real CSV data first
loader = CSVDataLoader()
df = loader.load_data()

if df is not None:
    print("[OK] Using REAL lap data from CSV")
    df = loader.prepare_for_ai_analysis(df)
    stats = loader.get_summary_statistics(df)
    print(f"  Sessions: {stats.get('num_sessions', 'N/A')}")
    print(f"  Total laps: {stats['total_laps']}")
    print(f"  Best lap: {stats['best_lap_time']:.3f}s")
    print(f"  Avg lap: {stats['average_lap_time']:.3f}s")
    using_real_data = True
else:
    print("⚠️  No real data found - Using mock data for demo")
    print("   To use real data: Export .ibt files to CSV")
    print("   See REAL_DATA_ANALYSIS.md for instructions")
    print()
    using_real_data = False

    # Generate mock data for demonstration
    df = generate_mock_data()
    print(f"   Generated {len(df)} sessions")
    print(f"   Lap time range: {df['fastest_time'].min():.3f}s - {df['fastest_time'].max():.3f}s")
    print()

# Step 1.25: Load session history for iterative learning
print("[1.25/5] Loading session memory...")
print()

session_mgr = SessionManager()
session_history = session_mgr.load_session_history(limit=5)
learning_metrics = session_mgr.get_learning_metrics()

if session_history:
    print(f"   [SESSION MEMORY] Loaded {len(session_history)} previous stint(s)")

    # Display session summary
    if learning_metrics:
        print(f"   [LEARNING] Total sessions tracked: {learning_metrics.get('total_sessions', 0)}")

        param_tests = learning_metrics.get('parameter_tests', {})
        if param_tests:
            most_tested = sorted(param_tests.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   [LEARNING] Most tested parameters:")
            for param, count in most_tested:
                print(f"      • {param}: {count}x")

        convergence = learning_metrics.get('convergence_metric')
        if convergence:
            print(f"   [LEARNING] Convergence: {convergence:.0%} (focus on consistent params)")

    # Show brief history
    print(f"\n   Recent stint history:")
    for i, session in enumerate(session_history[:3], 1):
        timestamp = session.get('timestamp', 'Unknown')[:16]  # Show date + time
        rec = session.get('recommendation', 'N/A')
        # Extract just the parameter name from recommendation
        param_name = "N/A"
        for word in rec.split():
            if 'tire_' in word or 'cross_' in word or 'spring_' in word or 'track_bar' in word:
                param_name = word.strip('.,;:')
                break
        print(f"      {i}. {timestamp} -> Tested {param_name}")

else:
    print("   [SESSION MEMORY] No previous stints found")
    print("   [INFO] This is your first stint - session learning will begin after this run")

print()

# Step 1.5: Gather driver feedback (INTERACTIVE)
print("[1.5/5] Driver Feedback Session...")
print()
print("DRIVER DEBRIEF:")
print("   (The crew chief asks the driver about the car's handling...)")
print()

# Check if running interactively or in batch mode
import sys
if len(sys.argv) > 1:
    # Command-line argument provided
    raw_driver_feedback = ' '.join(sys.argv[1:])
    print(f"   Driver: \"{raw_driver_feedback}\"")
else:
    # Interactive input
    print("   Enter driver feedback (or press Enter for default):")
    print("   Examples:")
    print("     - 'Car feels loose coming off the corners'")
    print("     - 'Front end pushes in turn 1 and 2'")
    print("     - 'Bottoming out in the center of corners'")
    print()
    print("   Driver: ", end='', flush=True)
    raw_driver_feedback = input().strip()

    if not raw_driver_feedback:
        # Use default example
        raw_driver_feedback = "The car feels a bit loose coming off the corners. I'm fighting oversteer in turns 1 and 2, especially on exit. Rear end wants to come around when I get on the throttle."
        print(f"\n   (Using default feedback: \"{raw_driver_feedback[:80]}...\")")

print()
print("   [AI] Interpreting driver feedback with AI...")

# Use LLM to interpret natural language feedback
from driver_feedback_interpreter import interpret_driver_feedback_with_llm

# Try to use LLM (falls back to rule-based if no API key)
driver_feedback = interpret_driver_feedback_with_llm(
    raw_driver_feedback,
    llm_provider="anthropic"  # or "openai" or "mock"
)

print(f"   [OK] Interpretation complete")
print(f"      Complaint type: {driver_feedback['complaint']}")
print(f"      Severity: {driver_feedback['severity']}")
print(f"      Technical diagnosis: {driver_feedback['diagnosis']}")
print()

# Step 2: Run full AI Race Engineer workflow (all agents)
print("[2/5] Running AI Race Engineer Workflow...")
print()

from race_engineer import app

initial_state = {
    'raw_setup_data': df,
    'driver_feedback': driver_feedback,
    'data_quality_decision': None,
    'analysis_strategy': None,
    'selected_features': None,
    'analysis': None,
    'recommendation': None,
    'error': None,
    # Session memory fields for iterative learning
    'session_history': session_history,
    'session_timestamp': datetime.now().isoformat(),
    'learning_metrics': learning_metrics,
    'previous_recommendations': [s.get('recommendation') for s in session_history[:3]] if session_history else None,
    'outcome_feedback': None,
    'convergence_progress': learning_metrics.get('convergence_metric') if learning_metrics else None
}

# Run the full LangGraph workflow (Telemetry Chief -> Data Scientist -> Crew Chief)
state = app.invoke(initial_state)

if 'error' in state and state['error']:
    print(f"   [ERROR] {state['error']}")
    sys.exit(1)

# Save session to persistent storage for iterative learning
session_id = session_mgr.save_session(state)
print(f"   [SESSION] Saved to session memory: {session_id}")

print()

# Step 3: Save results
print("[3/6] Saving results...")

results = {
    'data_source': 'real_csv_data' if using_real_data else 'mock_data',
    'recommendation': state.get('recommendation', 'No recommendation'),
    'analysis': state.get('analysis', {}),
    'best_time': float(df['fastest_time'].min()),
    'baseline_time': 15.543 if not using_real_data else float(df['fastest_time'].max()),
    'improvement': (15.543 if not using_real_data else float(df['fastest_time'].max())) - float(df['fastest_time'].min()),
    'num_sessions': len(df)
}

output_path = Path("output/demo_results.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"   Results saved to: {output_path}")
print()

# Step 4: Display summary
print("[4/6] Results Summary")
print("="*70)
print()
print("CREW CHIEF RECOMMENDATION:")
print(f"   {state.get('recommendation', 'No recommendation')}")
print()

analysis = state.get('analysis', {})
if analysis:
    print("KEY FINDINGS (Impact on Lap Time):")
    all_impacts = analysis.get('all_impacts', {})
    sorted_impacts = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    for param, impact in sorted_impacts[:5]:
        direction = "REDUCE" if impact > 0 else "INCREASE"
        print(f"   {param:30s}: {impact:+.3f}  [{direction}]")

print()
print("PERFORMANCE IMPROVEMENT:")
print(f"   Baseline time:  15.543s")
print(f"   Best AI time:   {df['fastest_time'].min():.3f}s")
print(f"   Improvement:    {15.543 - df['fastest_time'].min():.3f}s")
print()

# Step 5: Demo complete
print("="*70)
print("  [5/6] DEMO COMPLETE!")
print("="*70)
print()
print("Next steps:")
print("1. Review results in output/demo_results.json")
print("2. Run 'python create_visualizations.py' to generate charts")
print("3. Apply recommendations on track and validate")
