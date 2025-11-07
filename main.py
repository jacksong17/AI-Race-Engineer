"""
Bristol AI Race Engineer - Main Demo Application
Single entry point to run the complete demo from start to finish
"""

import sys
from pathlib import Path
import pandas as pd
import json
import numpy as np

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_section(text):
    """Print a section header"""
    print(f"\n{'-'*70}")
    print(f"  {text}")
    print(f"{'-'*70}")

def generate_mock_training_data(num_sessions=20):
    """Generate realistic mock telemetry data for demo purposes"""
    print_section("[DATA] Generating Mock Training Data")
    print(f"   Creating {num_sessions} simulated Bristol test sessions...")

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

    for i in range(num_sessions):
        session = baseline_setup.copy()
        session['session_id'] = f"bristol_test_{i+1}"

        # Vary parameters systematically
        if i < 5:
            # Baseline runs
            session['fastest_time'] = 15.543 + np.random.normal(0, 0.05)
        elif i < 8:
            # LF pressure tests
            session['tire_psi_lf'] = 28.0 + (i - 5) * 2
            # Lower LF pressure helps (slightly)
            session['fastest_time'] = 15.543 - 0.05 * (8 - i) + np.random.normal(0, 0.03)
        elif i < 11:
            # Cross weight tests
            session['cross_weight'] = 52.0 + (i - 8) * 2
            # Higher cross weight helps
            session['fastest_time'] = 15.543 - 0.08 * (i - 8) + np.random.normal(0, 0.03)
        elif i < 14:
            # Track bar tests
            session['track_bar_height_left'] = 5.0 + (i - 11) * 5
            session['fastest_time'] = 15.543 - 0.04 * (i - 11) + np.random.normal(0, 0.03)
        else:
            # Combined optimal settings
            session['tire_psi_lf'] = 26.0  # Lower is better
            session['cross_weight'] = 56.0  # Higher is better
            session['track_bar_height_left'] = 12.0
            # Best times with combined changes
            session['fastest_time'] = 15.543 - 0.30 + np.random.normal(0, 0.02)

        sessions.append(session)

    df = pd.DataFrame(sessions)

    # Save to bristol_data directory
    output_path = Path("bristol_data/mock_training_data.csv")
    df.to_csv(output_path, index=False)

    print(f"   [OK] Generated {len(df)} sessions")
    print(f"   [FILE] Saved to: {output_path}")
    print(f"   [TIME]  Lap time range: {df['fastest_time'].min():.3f}s - {df['fastest_time'].max():.3f}s")

    return df

def create_mock_ldx_data():
    """Create mock LDX data for the demo"""
    print_section("[SETUP] Creating Mock Setup Data")

    from telemetry_parser import TelemetryParser

    # Generate mock data that looks like parsed .ldx files
    df = generate_mock_training_data(20)

    # Save as individual "sessions" in the expected format
    parser = TelemetryParser()

    print(f"   [OK] Mock setup data ready for AI agents")
    return df

def test_ibt_parsing():
    """Test parsing the real Bristol .ibt file"""
    print_section("[TELEMETRY] Testing Real Telemetry Parsing")

    from ibt_parser import IBTParser

    ibt_parser = IBTParser()

    # Look for the Bristol .ibt file
    ibt_files = list(Path("data/raw/telemetry").glob("*.ibt"))

    if not ibt_files:
        print("   [WARN]  No .ibt files found in data/raw/telemetry/")
        print("   [INFO]  Using mock telemetry data for demo")
        return None

    print(f"   Found {len(ibt_files)} .ibt file(s)")

    # Parse the first one
    telemetry_df = ibt_parser.parse_ibt_file(ibt_files[0])
    lap_stats = ibt_parser.extract_lap_statistics(telemetry_df)

    if not lap_stats.empty:
        print(f"   [OK] Successfully parsed telemetry")
        print(f"   [DATA] Extracted {len(lap_stats)} laps")
        print(f"   [TIME]  Best lap: {lap_stats['lap_time'].min():.3f}s")
        print(f"   [CHART] Average: {lap_stats['lap_time'].mean():.3f}s")

        # Save lap statistics
        output_path = Path("output/lap_statistics.csv")
        lap_stats.to_csv(output_path, index=False)
        print(f"   [SAVE] Saved to: {output_path}")

    return lap_stats

def run_ai_agents():
    """Run the AI Race Engineer agents"""
    print_section("[AI] Running AI Race Engineer Agents")

    # Check if we have real data or need to use mock
    mock_data_path = Path("bristol_data/mock_training_data.csv")

    if not mock_data_path.exists():
        print("   Creating mock data for demo...")
        df = create_mock_ldx_data()
    else:
        print("   Loading existing training data...")
        df = pd.read_csv(mock_data_path)

    print(f"   [DATA] Data loaded: {len(df)} training sessions")

    # Instead of using the LangGraph workflow (which expects .ldx files),
    # let's run the agent logic directly on our DataFrame
    from race_engineer import analysis_agent, engineer_agent

    # Create a state-like dict for the agents
    state = {
        'raw_setup_data': df,
        'analysis': None,
        'recommendation': None,
        'error': None
    }

    # Run analysis agent
    print("\n   [ANALYSIS] Data Scientist Agent: Analyzing correlations...")
    analysis_result = analysis_agent(state)

    if 'error' in analysis_result and analysis_result['error']:
        print(f"   [ERROR] Error: {analysis_result['error']}")
        return None

    state.update(analysis_result)

    # Run engineer agent
    print("\n   [SETUP] Crew Chief Agent: Generating recommendations...")
    engineer_result = engineer_agent(state)

    if 'error' in engineer_result and engineer_result['error']:
        print(f"   [ERROR] Error: {engineer_result['error']}")
        return None

    state.update(engineer_result)

    return state

def create_visualizations():
    """Generate demo visualizations"""
    print_section("[VISUAL] Creating Visualizations")

    try:
        from create_visualizations import create_demo_visualizations

        print("   Generating dashboard and key insights...")
        create_demo_visualizations()
        print("   [OK] Visualizations created successfully!")
        print("   [FILE] Files created:")
        print("      - bristol_analysis_dashboard.png")
        print("      - bristol_key_insights.png")
    except Exception as e:
        print(f"   [WARN]  Could not create visualizations: {e}")
        print("   [INFO]  You can run create_visualizations.py separately")

def save_results(state):
    """Save the final results"""
    print_section("[SAVE] Saving Results")

    if not state:
        print("   [WARN]  No results to save")
        return

    # Save recommendation
    results = {
        'recommendation': state.get('recommendation', 'No recommendation generated'),
        'analysis': state.get('analysis', {}),
        'timestamp': pd.Timestamp.now().isoformat()
    }

    output_path = Path("output/race_engineer_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"   [OK] Results saved to: {output_path}")

def display_summary(state):
    """Display final summary"""
    print_banner("[FLAG] BRISTOL AI RACE ENGINEER - RESULTS SUMMARY")

    if not state:
        print("   [ERROR] No results available")
        return

    print("   [DATA] ANALYSIS COMPLETE")
    print()

    recommendation = state.get('recommendation', 'No recommendation')
    print("   [SETUP] CREW CHIEF RECOMMENDATION:")
    print(f"   {recommendation}")
    print()

    analysis = state.get('analysis', {})
    if analysis:
        print("   [CHART] KEY FINDINGS:")
        all_impacts = analysis.get('all_impacts', {})
        sorted_impacts = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

        for param, impact in sorted_impacts[:5]:
            direction = "REDUCE REDUCE" if impact > 0 else "INCREASE INCREASE"
            print(f"      {param:25s}: {impact:+.3f}  {direction}")

    print()
    print("   [INFO] NEXT STEPS:")
    print("      1. Apply recommended setup changes")
    print("      2. Test on track to validate predictions")
    print("      3. Collect new data with optimized setup")
    print("      4. Re-run analysis to find additional gains")
    print()

def main():
    """Main application entry point"""
    print_banner("[FLAG] BRISTOL AI RACE ENGINEER")
    print("   Complete Demo Application")
    print("   Optimizing NASCAR Truck Setup at Bristol Motor Speedway")

    try:
        # Step 1: Test real telemetry parsing
        test_ibt_parsing()

        # Step 2: Create/load training data
        if not Path("bristol_data/mock_training_data.csv").exists():
            create_mock_ldx_data()

        # Step 3: Run AI agents
        state = run_ai_agents()

        # Step 4: Create visualizations
        create_visualizations()

        # Step 5: Save results
        save_results(state)

        # Step 6: Display summary
        display_summary(state)

        print_banner("[OK] DEMO COMPLETE")
        print("   All outputs saved to 'output/' directory")
        print("   Visualizations ready for presentation")
        print()

    except KeyboardInterrupt:
        print("\n\n[ERROR] Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Error running demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
