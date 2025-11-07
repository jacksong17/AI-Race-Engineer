"""
Simplified Bristol AI Race Engineer Demo
Runs the core AI analysis without visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import json
import numpy as np

print("="*70)
print("  BRISTOL AI RACE ENGINEER - SIMPLIFIED DEMO")
print("="*70)
print()

# Step 1: Generate mock training data
print("[1/5] Generating mock training data...")
print()

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

df = pd.DataFrame(sessions)
print(f"   Generated {len(df)} sessions")
print(f"   Lap time range: {df['fastest_time'].min():.3f}s - {df['fastest_time'].max():.3f}s")
print()

# Step 2: Run analysis agent
print("[2/5] Running Data Scientist Agent...")
print()

from race_engineer import analysis_agent, engineer_agent

state = {
    'raw_setup_data': df,
    'analysis': None,
    'recommendation': None,
    'error': None
}

analysis_result = analysis_agent(state)

if 'error' in analysis_result and analysis_result['error']:
    print(f"   [ERROR] {analysis_result['error']}")
    sys.exit(1)

state.update(analysis_result)
print()

# Step 3: Run engineer agent
print("[3/5] Running Crew Chief Agent...")
print()

engineer_result = engineer_agent(state)

if 'error' in engineer_result and engineer_result['error']:
    print(f"   [ERROR] {engineer_result['error']}")
    sys.exit(1)

state.update(engineer_result)
print()

# Step 4: Save results
print("[4/5] Saving results...")

results = {
    'recommendation': state.get('recommendation', 'No recommendation'),
    'analysis': state.get('analysis', {}),
    'best_time': float(df['fastest_time'].min()),
    'baseline_time': 15.543,
    'improvement': 15.543 - float(df['fastest_time'].min()),
    'num_sessions': len(df)
}

output_path = Path("output/demo_results.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"   Results saved to: {output_path}")
print()

# Step 5: Display summary
print("[5/5] Results Summary")
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

print("="*70)
print("  DEMO COMPLETE!")
print("="*70)
print()
print("Next steps:")
print("1. Review results in output/demo_results.json")
print("2. Run 'python create_visualizations.py' to generate charts")
print("3. Apply recommendations on track and validate")
