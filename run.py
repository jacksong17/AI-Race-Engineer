"""
Bristol AI Race Engineer - Concise Output
Runs analysis with minimal output - just the actionable recommendations
"""

import sys
from pathlib import Path
import pandas as pd
import json
import os

# Suppress verbose logging
os.environ['QUIET_MODE'] = '1'

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from csv_data_loader import CSVDataLoader
from session_manager import SessionManager
from driver_feedback_interpreter import interpret_driver_feedback_with_llm
from constraint_extractor import extract_constraints
from knowledge_base_loader import load_setup_manual
from race_engineer import app
from datetime import datetime

print("="*70)
print("  BRISTOL AI RACE ENGINEER")
print("="*70)
print()

# Load data
loader = CSVDataLoader()
df = loader.load_data()

if df is None:
    print("⚠️  No real data found. Please add CSV data to run analysis.")
    sys.exit(1)

df = loader.prepare_for_ai_analysis(df)
stats = loader.get_summary_statistics(df)

print(f"Loaded {stats['total_laps']} laps (Best: {stats['best_lap_time']:.3f}s)")
print()

# Load session history
session_mgr = SessionManager()
session_history = session_mgr.load_session_history(limit=5)
learning_metrics = session_mgr.get_learning_metrics()

if session_history:
    print(f"Session memory: {len(session_history)} previous run(s) loaded")
print()

# Get driver feedback
print("DRIVER FEEDBACK:")
if len(sys.argv) > 1:
    raw_driver_feedback = ' '.join(sys.argv[1:])
    print(f'  "{raw_driver_feedback}"')
else:
    print("  Enter feedback: ", end='', flush=True)
    raw_driver_feedback = input().strip()
    if not raw_driver_feedback:
        raw_driver_feedback = "The car feels a bit loose coming off the corners."
        print(f'  (Using default: "{raw_driver_feedback}")')

print()

# Interpret feedback
driver_feedback = interpret_driver_feedback_with_llm(
    raw_driver_feedback,
    llm_provider="anthropic",
    multi_issue=False
)

print(f"  Diagnosis: {driver_feedback['diagnosis']}")
print(f"  Severity: {driver_feedback['severity']}")
print()

# Extract constraints
driver_constraints = extract_constraints(raw_driver_feedback, llm_provider="anthropic")
if driver_constraints and any([
    driver_constraints.get('parameter_limits'),
    driver_constraints.get('already_tried'),
    driver_constraints.get('cannot_adjust')
]):
    print("  Constraints:")
    for limit in driver_constraints.get('parameter_limits', []):
        print(f"    • {limit['param']}: {limit['limit_type']}")
    if driver_constraints.get('already_tried'):
        print(f"    • Already tried: {', '.join(driver_constraints['already_tried'])}")
    print()

# Load setup knowledge
setup_knowledge_base = load_setup_manual()

# Run analysis
print("Running analysis...")
initial_state = {
    'raw_setup_data': df,
    'driver_feedback': driver_feedback,
    'data_quality_decision': None,
    'analysis_strategy': None,
    'selected_features': None,
    'analysis': None,
    'recommendation': None,
    'error': None,
    'session_history': session_history,
    'session_timestamp': datetime.now().isoformat(),
    'learning_metrics': learning_metrics,
    'previous_recommendations': [s.get('recommendation') for s in session_history[:3]] if session_history else None,
    'outcome_feedback': None,
    'convergence_progress': learning_metrics.get('convergence_metric') if learning_metrics else None,
    'raw_driver_feedback': raw_driver_feedback,
    'driver_constraints': driver_constraints,
    'setup_knowledge_base': setup_knowledge_base
}

state = app.invoke(initial_state)

if 'error' in state and state['error']:
    print(f"⚠️  Error: {state['error']}")
    sys.exit(1)

print()
print("="*70)
print("  RECOMMENDATION")
print("="*70)
print()
print(state.get('recommendation', 'No recommendation'))
print()

# Show key findings
analysis = state.get('analysis', {})
if analysis:
    print("="*70)
    print("  TOP 5 CORRELATIONS")
    print("="*70)
    print()
    all_impacts = analysis.get('all_impacts', {})
    sorted_impacts = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    for param, impact in sorted_impacts[:5]:
        direction = "REDUCE" if impact > 0 else "INCREASE"
        print(f"  {direction:8s} {param:25s} (correlation: {impact:+.3f})")
    print()

# Save session
session_id = session_mgr.save_session(
    driver_feedback=raw_driver_feedback,
    recommendation=state.get('recommendation'),
    analysis_results=state.get('analysis'),
    metadata={'data_source': 'real_csv_data'}
)

print(f"Session saved: {session_id}")
print()
print("="*70)
print("  COMPLETE")
print("="*70)
