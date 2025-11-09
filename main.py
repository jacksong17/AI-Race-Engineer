#!/usr/bin/env python3
"""
AI Race Engineer - Main CLI Interface

Production-ready agentic system for NASCAR racing telemetry analysis.
"""

import sys
import argparse
from pathlib import Path
import json
from typing import Optional, List

# Add race_engineer to path
sys.path.insert(0, str(Path(__file__).parent))

from race_engineer.graph import (
    create_race_engineer_graph,
    create_checkpointer,
    run_workflow,
    print_graph_info
)
from race_engineer.state import create_initial_state


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI Race Engineer - Agentic NASCAR Telemetry Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with driver feedback
  python main.py --feedback "Car is loose off turn 2, rear end slides"

  # Specify telemetry files
  python main.py --feedback "Tight on entry" --telemetry "data/telemetry/*.csv"

  # With constraints
  python main.py --feedback "Pushing in corners" --constraints constraints.json

  # Show graph structure
  python main.py --show-graph

  # Verbose output
  python main.py --feedback "Oversteer on exit" --verbose
        """
    )

    parser.add_argument(
        "--feedback",
        type=str,
        help="Driver feedback text describing handling issues"
    )

    parser.add_argument(
        "--telemetry",
        type=str,
        default="data/processed/*.csv",
        help="Path to telemetry files (supports glob patterns)"
    )

    parser.add_argument(
        "--constraints",
        type=str,
        help="Path to JSON file with driver constraints"
    )

    parser.add_argument(
        "--track",
        type=str,
        default="bristol",
        help="Track name (default: bristol)"
    )

    parser.add_argument(
        "--car",
        type=str,
        default="nascar_truck",
        help="Car class (default: nascar_truck)"
    )

    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Display graph structure and exit"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output with detailed progress"
    )

    parser.add_argument(
        "--save-session",
        action="store_true",
        help="Save session results to output directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output/sessions",
        help="Output directory for results (default: output/sessions)"
    )

    return parser.parse_args()


def load_constraints(filepath: str) -> Optional[dict]:
    """Load driver constraints from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Warning: Could not load constraints from {filepath}: {e}")
        return None


def get_telemetry_files(pattern: str) -> List[str]:
    """Get list of telemetry files matching pattern"""
    files = list(Path().glob(pattern))
    if not files:
        print(f"⚠️  No telemetry files found matching: {pattern}")
        print("    Will use mock data for demonstration")
    return [str(f) for f in files]


def display_results(state: dict, verbose: bool = False):
    """Display final results in a formatted way"""
    print("\n" + "="*70)
    print("📋 FINAL RESULTS")
    print("="*70)

    # Recommendation
    if state.get('final_recommendation'):
        rec = state['final_recommendation']
        print("\n💡 RECOMMENDATION:")
        if 'summary' in rec:
            print(rec['summary'])
        elif 'primary' in rec and rec['primary']:
            primary = rec['primary']
            print(f"\n  Primary Change:")
            print(f"    • {primary['parameter']}")
            print(f"    • {primary['direction'].title()} by {primary['magnitude']} {primary['magnitude_unit']}")
            print(f"    • Rationale: {primary['rationale']}")
            print(f"    • Confidence: {primary['confidence']:.1%}")

    # Data quality
    if state.get('data_quality_report') and verbose:
        qr = state['data_quality_report']
        print(f"\n📊 DATA QUALITY:")
        print(f"  • Sessions analyzed: {qr.get('num_sessions', 'N/A')}")
        print(f"  • Quality score: {qr.get('quality_score', 0):.1%}")
        print(f"  • Usable parameters: {len(qr.get('usable_parameters', []))}")

    # Analysis method
    if state.get('statistical_analysis'):
        stats = state['statistical_analysis']
        print(f"\n📈 ANALYSIS METHOD: {stats.get('method', 'unknown').title()}")
        if 'r_squared' in stats:
            print(f"  • Model R²: {stats['r_squared']:.3f}")

    # Visualizations
    if state.get('generated_visualizations'):
        print(f"\n📊 VISUALIZATIONS:")
        for viz in state['generated_visualizations']:
            print(f"  • {viz}")

    # Warnings
    if state.get('warnings'):
        print(f"\n⚠️  WARNINGS:")
        for warning in state['warnings']:
            print(f"  • {warning}")

    # Errors
    if state.get('errors'):
        print(f"\n❌ ERRORS:")
        for error in state['errors']:
            print(f"  • {error.get('message', str(error))}")

    # Workflow stats
    if verbose:
        print(f"\n🔄 WORKFLOW STATS:")
        print(f"  • Iterations: {state.get('iteration', 0)}")
        print(f"  • Agents consulted: {', '.join(state.get('agents_consulted', []))}")
        print(f"  • Tools called: {len(state.get('tools_called', []))}")

    print("\n" + "="*70)


def save_session_results(state: dict, output_dir: str):
    """Save session results to JSON file"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        session_id = state.get('session_id', 'unknown')
        filename = f"session_{session_id}.json"
        filepath = output_path / filename

        # Prepare serializable state
        save_state = {
            "session_id": state.get('session_id'),
            "session_timestamp": state.get('session_timestamp'),
            "driver_feedback": state.get('driver_feedback'),
            "final_recommendation": state.get('final_recommendation'),
            "statistical_analysis": state.get('statistical_analysis'),
            "data_quality_report": state.get('data_quality_report'),
            "warnings": state.get('warnings', []),
            "errors": state.get('errors', []),
            "iteration": state.get('iteration'),
            "agents_consulted": state.get('agents_consulted', []),
            "generated_visualizations": state.get('generated_visualizations', [])
        }

        with open(filepath, 'w') as f:
            json.dump(save_state, f, indent=2, default=str)

        print(f"\n💾 Session saved to: {filepath}")

    except Exception as e:
        print(f"\n⚠️  Could not save session: {e}")


def main():
    """Main entry point"""
    args = parse_arguments()

    # Create the graph
    print("\n🏗️  Building agentic workflow graph...")
    app = create_race_engineer_graph()

    # Show graph structure if requested
    if args.show_graph:
        print_graph_info(app)
        try:
            mermaid = app.get_graph().draw_mermaid()
            print("\nMermaid Diagram:")
            print(mermaid)
        except Exception as e:
            print(f"Could not generate diagram: {e}")
        sys.exit(0)

    # Validate feedback is provided
    if not args.feedback:
        print("❌ Error: --feedback is required")
        print("Example: python main.py --feedback 'Car is loose on corner exit'")
        sys.exit(1)

    # Load constraints if provided
    constraints = None
    if args.constraints:
        constraints = load_constraints(args.constraints)

    # Get telemetry files
    telemetry_files = get_telemetry_files(args.telemetry)

    # Session config
    config = {
        "track": args.track,
        "car_class": args.car,
        "conditions": "dry",
        "analysis_mode": "standard"
    }

    # Run the workflow
    print("\n🚀 Starting analysis...")

    try:
        final_state = run_workflow(
            app=app,
            driver_feedback=args.feedback,
            telemetry_files=telemetry_files,
            constraints=constraints,
            config=config,
            verbose=args.verbose
        )

        # Display results
        display_results(final_state, verbose=args.verbose)

        # Save session if requested
        if args.save_session:
            save_session_results(final_state, args.output)

        # Exit with success
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n❌ Analysis failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
