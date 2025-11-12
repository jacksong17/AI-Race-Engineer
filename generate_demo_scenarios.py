"""
Generate comprehensive demo scenarios for AI Race Engineer.

Creates realistic test cases that showcase analytical capabilities:
- Statistical rigor (confidence intervals, effect sizes, interactions)
- Multi-agent coordination
- Edge cases (small samples, conflicts, quality gates)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


def generate_telemetry_sessions(
    base_params: dict,
    target_time: float,
    n_sessions: int,
    varied_param: str,
    param_range: tuple,
    effect_strength: float = -0.1
) -> list:
    """
    Generate realistic telemetry session data with controlled variation.

    Args:
        base_params: Baseline parameter values
        target_time: Baseline lap time
        n_sessions: Number of sessions to generate
        varied_param: Parameter to vary systematically
        param_range: (min, max) range for varied parameter
        effect_strength: Impact on lap time (negative = faster)

    Returns:
        List of session dictionaries
    """
    sessions = []
    np.random.seed(42)  # Reproducible

    # Generate systematic variation
    param_values = np.linspace(param_range[0], param_range[1], n_sessions)

    for i, param_val in enumerate(param_values):
        # Calculate lap time based on parameter
        param_effect = (param_val - base_params[varied_param]) * effect_strength
        noise = np.random.normal(0, 0.05)  # Natural variance
        lap_time = target_time + param_effect + noise

        # Build session
        session = base_params.copy()
        session['session_id'] = f"session_{i+1:02d}"
        session['fastest_time'] = round(lap_time, 3)
        session[varied_param] = round(param_val, 2)

        # Add small random variations to other parameters
        for key in session:
            if key not in ['session_id', 'fastest_time', varied_param]:
                if isinstance(session[key], (int, float)):
                    session[key] = round(session[key] + np.random.normal(0, 0.5), 2)

        sessions.append(session)

    return sessions


def create_scenario_1_understeer():
    """Scenario 1: Classic understeer - front tire pressure solution"""

    base_params = {
        'tire_psi_lf': 32.0,
        'tire_psi_rf': 32.0,
        'tire_psi_lr': 31.0,
        'tire_psi_rr': 31.0,
        'cross_weight': 52.5,
        'spring_lf': 950,
        'spring_rf': 950,
        'spring_lr': 200,
        'spring_rr': 200,
        'track_bar_height_left': 12.0,
        'brake_bias': 58.5
    }

    # Generate 15 sessions varying front right tire pressure
    sessions = generate_telemetry_sessions(
        base_params=base_params,
        target_time=15.35,
        n_sessions=15,
        varied_param='tire_psi_rf',
        param_range=(29.0, 34.0),
        effect_strength=-0.15  # Lowering pressure improves grip -> faster
    )

    return {
        "metadata": {
            "scenario_name": "Understeer on Corner Entry",
            "track": "bristol",
            "car_class": "nascar_truck",
            "difficulty": "moderate",
            "expected_iterations": 3
        },
        "input": {
            "driver_feedback": "Car won't turn in on corner entry. Really pushing through turns 1 and 3. Front end feels tight.",
            "telemetry_sessions": sessions,
            "constraints": None
        },
        "expected_output": {
            "top_recommendation": "tire_psi_rf",
            "direction": "decrease",
            "magnitude_range": [1.0, 2.5],
            "confidence_min": 0.70,
            "quality_gate": "pass",
            "statistical_method": "correlation",
            "interaction_detected": False
        }
    }


def create_scenario_2_oversteer():
    """Scenario 2: Oversteer on exit - rear spring adjustment"""

    base_params = {
        'tire_psi_lf': 31.0,
        'tire_psi_rf': 31.0,
        'tire_psi_lr': 30.0,
        'tire_psi_rr': 30.0,
        'cross_weight': 52.0,
        'spring_lf': 900,
        'spring_rf': 900,
        'spring_lr': 180,
        'spring_rr': 180,
        'track_bar_height_left': 12.5,
        'brake_bias': 57.0
    }

    # Vary rear right spring - increasing helps oversteer
    sessions = generate_telemetry_sessions(
        base_params=base_params,
        target_time=15.45,
        n_sessions=12,
        varied_param='spring_rr',
        param_range=(170, 220),
        effect_strength=-0.08  # Stiffer rear spring reduces oversteer
    )

    return {
        "metadata": {
            "scenario_name": "Oversteer on Corner Exit",
            "track": "bristol",
            "car_class": "nascar_truck",
            "difficulty": "moderate",
            "expected_iterations": 3
        },
        "input": {
            "driver_feedback": "Car is really loose getting back to throttle. Tail wants to come around in turn 2.",
            "telemetry_sessions": sessions,
            "constraints": None
        },
        "expected_output": {
            "top_recommendation": "spring_rr",
            "direction": "increase",
            "magnitude_range": [15, 30],
            "confidence_min": 0.65,
            "quality_gate": "pass",
            "statistical_method": "correlation"
        }
    }


def create_scenario_3_multi_issue():
    """Scenario 3: Multiple interacting issues requiring iteration"""

    base_params = {
        'tire_psi_lf': 33.0,
        'tire_psi_rf': 33.0,
        'tire_psi_lr': 31.0,
        'tire_psi_rr': 31.0,
        'cross_weight': 54.0,
        'spring_lf': 1000,
        'spring_rf': 1000,
        'spring_lr': 220,
        'spring_rr': 220,
        'track_bar_height_left': 11.5,
        'brake_bias': 60.0
    }

    # Create dataset with both tire pressure AND cross-weight effects
    sessions = []
    np.random.seed(123)

    for i in range(18):
        session = base_params.copy()
        session['session_id'] = f"session_{i+1:02d}"

        # Vary both parameters
        session['tire_psi_rf'] = round(29 + i * 0.3, 1)
        session['cross_weight'] = round(52 + i * 0.15, 2)

        # Combined effect (interaction)
        tire_effect = (session['tire_psi_rf'] - 33.0) * -0.12
        weight_effect = (session['cross_weight'] - 54.0) * -0.05
        interaction_effect = ((session['tire_psi_rf'] - 31) * (session['cross_weight'] - 53)) * 0.01

        session['fastest_time'] = round(
            15.60 + tire_effect + weight_effect + interaction_effect + np.random.normal(0, 0.04),
            3
        )

        sessions.append(session)

    return {
        "metadata": {
            "scenario_name": "Complex Multi-Parameter Issue",
            "track": "bristol",
            "car_class": "nascar_truck",
            "difficulty": "hard",
            "expected_iterations": 4
        },
        "input": {
            "driver_feedback": "Car is tight in center but also unbalanced. Front push on entry, feels heavy overall.",
            "telemetry_sessions": sessions,
            "constraints": None
        },
        "expected_output": {
            "top_recommendation": ["tire_psi_rf", "cross_weight"],
            "quality_gate": "pass",
            "interaction_detected": True,
            "r2_gain_from_interactions": 0.10
        }
    }


def create_scenario_4_edge_case_small_sample():
    """Scenario 4: Edge case - insufficient data quality gate"""

    base_params = {
        'tire_psi_lf': 31.0,
        'tire_psi_rf': 31.0,
        'tire_psi_lr': 30.0,
        'tire_psi_rr': 30.0,
        'cross_weight': 52.0,
        'spring_lf': 920,
        'spring_rf': 920,
        'spring_lr': 195,
        'spring_rr': 195,
        'track_bar_height_left': 12.2,
        'brake_bias': 58.0
    }

    # Only 5 sessions - borderline quality
    sessions = generate_telemetry_sessions(
        base_params=base_params,
        target_time=15.40,
        n_sessions=5,
        varied_param='tire_psi_lf',
        param_range=(29.0, 32.0),
        effect_strength=-0.10
    )

    return {
        "metadata": {
            "scenario_name": "Insufficient Data Quality Check",
            "track": "bristol",
            "car_class": "nascar_truck",
            "difficulty": "edge_case",
            "expected_iterations": 2
        },
        "input": {
            "driver_feedback": "Bit of understeer but haven't tested much yet.",
            "telemetry_sessions": sessions,
            "constraints": None
        },
        "expected_output": {
            "quality_gate": "warning",
            "quality_score_max": 0.70,
            "confidence_max": 0.60,
            "recommendations": ["Collect more data sessions"]
        }
    }


def create_scenario_5_constraint_handling():
    """Scenario 5: Parameter at limit - needs alternative"""

    base_params = {
        'tire_psi_lf': 29.0,  # Already at minimum!
        'tire_psi_rf': 29.0,
        'tire_psi_lr': 30.0,
        'tire_psi_rr': 30.0,
        'cross_weight': 52.5,
        'spring_lf': 950,
        'spring_rf': 950,
        'spring_lr': 200,
        'spring_rr': 200,
        'track_bar_height_left': 12.0,
        'brake_bias': 58.5
    }

    # Data shows front tire pressure matters but it's at limit
    sessions = generate_telemetry_sessions(
        base_params=base_params,
        target_time=15.50,
        n_sessions=13,
        varied_param='cross_weight',
        param_range=(50.0, 55.0),
        effect_strength=-0.08
    )

    return {
        "metadata": {
            "scenario_name": "Constraint Handling - Parameter at Limit",
            "track": "bristol",
            "car_class": "nascar_truck",
            "difficulty": "moderate",
            "expected_iterations": 3
        },
        "input": {
            "driver_feedback": "Still tight on entry even after dropping tire pressure to minimum.",
            "telemetry_sessions": sessions,
            "constraints": {
                "parameters_at_limit": {
                    "tire_psi_lf": "min",
                    "tire_psi_rf": "min"
                },
                "already_tried": ["tire_psi_lf", "tire_psi_rf"]
            }
        },
        "expected_output": {
            "top_recommendation": "cross_weight",
            "direction": "decrease",
            "constraint_acknowledged": True,
            "alternative_found": True,
            "quality_gate": "pass"
        }
    }


def create_scenario_6_brake_issue():
    """Scenario 6: Brake lockup - brake bias adjustment"""

    base_params = {
        'tire_psi_lf': 31.0,
        'tire_psi_rf': 31.0,
        'tire_psi_lr': 30.0,
        'tire_psi_rr': 30.0,
        'cross_weight': 52.0,
        'spring_lf': 920,
        'spring_rf': 920,
        'spring_lr': 195,
        'spring_rr': 195,
        'track_bar_height_left': 12.2,
        'brake_bias': 62.0  # Too much front bias
    }

    sessions = generate_telemetry_sessions(
        base_params=base_params,
        target_time=15.42,
        n_sessions=14,
        varied_param='brake_bias',
        param_range=(56.0, 62.0),
        effect_strength=0.10  # Higher bias = slower (locking fronts)
    )

    return {
        "metadata": {
            "scenario_name": "Brake Lockup Issue",
            "track": "bristol",
            "car_class": "nascar_truck",
            "difficulty": "easy",
            "expected_iterations": 2
        },
        "input": {
            "driver_feedback": "Front brakes lock up easily, especially into turn 1. Can't brake late.",
            "telemetry_sessions": sessions,
            "constraints": None
        },
        "expected_output": {
            "top_recommendation": "brake_bias",
            "direction": "decrease",
            "magnitude_range": [2.0, 4.0],
            "confidence_min": 0.75,
            "quality_gate": "pass",
            "knowledge_base_match": True
        }
    }


def main():
    """Generate all demo scenarios and save to files"""

    output_dir = Path("demo_datasets")
    output_dir.mkdir(exist_ok=True)

    scenarios = [
        ("scenario_01_understeer_front_grip.json", create_scenario_1_understeer()),
        ("scenario_02_oversteer_rear_spring.json", create_scenario_2_oversteer()),
        ("scenario_03_multi_issue_complex.json", create_scenario_3_multi_issue()),
        ("scenario_04_edge_case_small_sample.json", create_scenario_4_edge_case_small_sample()),
        ("scenario_05_constraint_at_limit.json", create_scenario_5_constraint_handling()),
        ("scenario_06_brake_lockup_bias.json", create_scenario_6_brake_issue())
    ]

    print("Generating Demo Scenarios")
    print("=" * 60)

    for filename, scenario in scenarios:
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(scenario, f, indent=2)

        n_sessions = len(scenario['input']['telemetry_sessions'])
        print(f"✓ {filename}")
        print(f"  - {n_sessions} sessions")
        print(f"  - Difficulty: {scenario['metadata']['difficulty']}")
        print(f"  - Expected iterations: {scenario['metadata']['expected_iterations']}")
        print()

    # Generate summary manifest
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_scenarios": len(scenarios),
        "scenarios": [
            {
                "filename": fname,
                "name": scen['metadata']['scenario_name'],
                "difficulty": scen['metadata']['difficulty'],
                "sessions": len(scen['input']['telemetry_sessions'])
            }
            for fname, scen in scenarios
        ]
    }

    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print("=" * 60)
    print(f"✅ Generated {len(scenarios)} demo scenarios")
    print(f"📁 Saved to: {output_dir.absolute()}")
    print("\nUse these scenarios to:")
    print("  • Demo the AI Race Engineer capabilities")
    print("  • Regression testing after changes")
    print("  • Performance benchmarking (tokens, time, cost)")
    print("  • Validate statistical enhancements (CIs, effect sizes, interactions)")


if __name__ == "__main__":
    main()
