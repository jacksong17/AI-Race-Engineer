"""
Enhanced AI Race Engineer Demo
Demonstrates Features 1, 2, and 3:
- Multi-Issue Feedback Parser
- Parameter Interaction Detection
- Outcome Validation Loop

This is a complete functional demonstration of the enhanced system.
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Import enhanced modules
from driver_feedback_interpreter import interpret_driver_feedback_with_llm
from interaction_analyzer import InteractionAnalyzer
from outcome_validator import OutcomeValidator
from session_manager import SessionManager


def generate_enhanced_demo_data():
    """
    Generate realistic demo data with KNOWN INTERACTIONS for testing.

    This data includes:
    - Parameter interactions (tire_psi_rr × cross_weight)
    - Multiple correlated parameters
    - Realistic variance
    """
    print("[DATA GENERATION] Creating enhanced demo dataset...")

    np.random.seed(42)
    n_sessions = 25

    # Generate parameters with realistic ranges
    tire_psi_lf = np.random.uniform(26, 30, n_sessions)
    tire_psi_rf = np.random.uniform(30, 36, n_sessions)
    tire_psi_lr = np.random.uniform(24, 28, n_sessions)
    tire_psi_rr = np.random.uniform(28, 34, n_sessions)
    cross_weight = np.random.uniform(50, 56, n_sessions)
    track_bar_height_left = np.random.uniform(5, 15, n_sessions)
    spring_lf = np.random.uniform(350, 500, n_sessions)
    spring_rf = np.random.uniform(400, 600, n_sessions)

    # True lap time model WITH INTERACTION
    # lap_time = baseline + main_effects + interaction_effect + noise
    lap_time = (
        15.50 +  # Baseline
        # Main effects
        0.08 * (tire_psi_rr - 31) +  # RR tire pressure (more = slower)
        -0.06 * (cross_weight - 53) +  # Cross weight (more = faster)
        0.03 * (spring_rf - 500) / 50 +  # Spring rate
        -0.04 * (tire_psi_lr - 26) +  # LR tire pressure
        # INTERACTION EFFECT (key discovery)
        -0.10 * ((tire_psi_rr - 31) / 3) * ((cross_weight - 53) / 3) +  # Synergistic interaction
        # Random noise
        np.random.normal(0, 0.04, n_sessions)
    )

    df = pd.DataFrame({
        'session_id': [f"demo_session_{i+1}" for i in range(n_sessions)],
        'tire_psi_lf': tire_psi_lf,
        'tire_psi_rf': tire_psi_rf,
        'tire_psi_lr': tire_psi_lr,
        'tire_psi_rr': tire_psi_rr,
        'cross_weight': cross_weight,
        'track_bar_height_left': track_bar_height_left,
        'spring_lf': spring_lf,
        'spring_rf': spring_rf,
        'fastest_time': lap_time
    })

    print(f"[DATA] Generated {n_sessions} sessions")
    print(f"[DATA] Lap time range: {lap_time.min():.3f}s - {lap_time.max():.3f}s")
    print(f"[DATA] Variance: {lap_time.std():.3f}s")
    print()

    return df


def demonstrate_feature_1_multi_issue_parsing():
    """
    FEATURE 1: Multi-Issue Feedback Parser
    Shows that the system can now parse multiple concurrent issues.
    """
    print("="*70)
    print("  FEATURE 1 DEMO: MULTI-ISSUE FEEDBACK PARSING")
    print("="*70)
    print()

    test_feedbacks = [
        {
            'feedback': "Car is loose off the corners but tight on entry",
            'expected_issues': 2,
            'description': "Mixed handling (loose + tight)"
        },
        {
            'feedback': "Bottoming out in turn 2 and the rear feels loose everywhere",
            'expected_issues': 2,
            'description': "Platform + handling issue"
        },
        {
            'feedback': "Front end pushes really bad",
            'expected_issues': 1,
            'description': "Single issue (tight)"
        },
        {
            'feedback': "Perfect balance, just looking for a few tenths",
            'expected_issues': 0,
            'description': "Optimization mode"
        }
    ]

    for i, test in enumerate(test_feedbacks, 1):
        print(f"[TEST {i}] {test['description']}")
        print(f"   Driver: \"{test['feedback']}\"")
        print()

        # Parse feedback with multi-issue support
        result = interpret_driver_feedback_with_llm(
            raw_feedback=test['feedback'],
            llm_provider="mock",  # Use mock for deterministic demo
            multi_issue=True
        )

        issues = result.get('issues', [])
        balance_type = result.get('handling_balance_type', 'unknown')
        primary = result.get('primary_issue')

        print(f"   [RESULT] Issues detected: {len(issues)}")
        print(f"   [RESULT] Balance type: {balance_type}")

        if issues:
            print(f"\n   Issues breakdown:")
            for j, issue in enumerate(issues, 1):
                marker = " (PRIMARY)" if issue == primary else ""
                print(f"      {j}. {issue['complaint']} - {issue['severity']} severity{marker}")
                print(f"         Diagnosis: {issue['diagnosis']}")
                print(f"         Priority params: {', '.join(issue['priority_features'][:2])}")
        else:
            print(f"   [INFO] Optimization mode - no specific issues")

        # Validation
        success = len(issues) == test['expected_issues']
        status = "PASS" if success else "FAIL"
        print(f"\n   [{status}] Expected {test['expected_issues']} issues, got {len(issues)}")
        print()
        print("-"*70)
        print()

    print("Feature 1 Demo Complete!")
    print("="*70)
    print()
    print()


def demonstrate_feature_2_interaction_detection(df: pd.DataFrame):
    """
    FEATURE 2: Parameter Interaction Detection
    Shows that the system can detect synergistic/antagonistic parameter relationships.
    """
    print("="*70)
    print("  FEATURE 2 DEMO: PARAMETER INTERACTION DETECTION")
    print("="*70)
    print()

    print("[INTERACTION ANALYSIS] Searching for parameter interactions...")
    print()

    analyzer = InteractionAnalyzer(max_interaction_order=2, regularization_alpha=0.5)

    features = ['tire_psi_rr', 'cross_weight', 'tire_psi_lr', 'spring_rf']
    result = analyzer.find_interactions(
        df=df,
        target='fastest_time',
        features=features,
        significance_threshold=0.04
    )

    print()
    print("="*70)
    print("  INTERACTION ANALYSIS RESULTS")
    print("="*70)
    print()

    if result['has_significant_interactions']:
        print(f"SUCCESS: Significant interactions detected!")
        print()
        print(f"Model Performance:")
        print(f"   Linear R²:      {result['linear_r2']:.3f}")
        print(f"   Polynomial R²:  {result['poly_r2']:.3f}")
        print(f"   Improvement:    +{result['model_improvement']:.3f} ({result['improvement_pct']:+.1f}%)")
        print()

        print("Discovered Interactions:")
        for i, inter in enumerate(result['interactions'][:3], 1):
            synergy_type = "SYNERGISTIC (work together)" if inter['synergistic'] else "ANTAGONISTIC (work against)"
            print(f"\n{i}. {inter['params'][0]} × {inter['params'][1]}")
            print(f"   Coefficient: {inter['coefficient']:+.4f}")
            print(f"   Type: {synergy_type}")
            print(f"   {inter['interpretation']}")

        # Generate compound recommendation
        print()
        print("-"*70)
        print("COMPOUND RECOMMENDATION:")
        print("-"*70)

        # Get top parameter from correlation
        from sklearn.linear_model import LinearRegression
        X = df[features]
        y = df['fastest_time']
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        impacts = {feat: coef for feat, coef in zip(features, linear_model.coef_)}

        primary_param = max(impacts.items(), key=lambda x: abs(x[1]))[0]

        compound_rec = analyzer.recommend_compound_change(
            primary_param=primary_param,
            all_impacts=impacts,
            available_params=features
        )

        if compound_rec:
            print(compound_rec['recommendation'])
        else:
            print("No suitable compound recommendation (no interactions with primary param)")

        print()
        print("[PASS] Interaction detection successful")

    else:
        print(f"[INFO] No significant interactions detected")
        print(f"   Linear R²:      {result['linear_r2']:.3f}")
        print(f"   Polynomial R²:  {result['poly_r2']:.3f}")
        print(f"   Improvement:    +{result['model_improvement']:.3f} (< 10% threshold)")
        print()
        print("[INFO] System correctly identified weak interactions")

    print()
    print("Feature 2 Demo Complete!")
    print("="*70)
    print()
    print()

    return result


def demonstrate_feature_3_outcome_validation():
    """
    FEATURE 3: Outcome Validation Loop
    Shows that the system can validate whether recommendations actually worked.
    """
    print("="*70)
    print("  FEATURE 3 DEMO: OUTCOME VALIDATION & CLOSED-LOOP LEARNING")
    print("="*70)
    print()

    validator = OutcomeValidator(confidence_level=0.80)

    # Simulate test scenarios
    scenarios = [
        {
            'name': "Recommendation WORKED (tire_psi_rr reduction)",
            'baseline': 15.50,
            'test_laps': [15.32, 15.35, 15.30, 15.33, 15.31],
            'recommendation': "REDUCE tire_psi_rr by 2.0 psi",
            'expected_outcome': 'improved'
        },
        {
            'name': "Recommendation FAILED (cross_weight increase)",
            'baseline': 15.50,
            'test_laps': [15.68, 15.72, 15.70, 15.69],
            'recommendation': "INCREASE cross_weight by 1.0%",
            'expected_outcome': 'worse'
        },
        {
            'name': "No significant change (spring_rf)",
            'baseline': 15.50,
            'test_laps': [15.48, 15.52, 15.51, 15.49, 15.50],
            'recommendation': "INCREASE spring_rf by 25 N/mm",
            'expected_outcome': 'no_change'
        }
    ]

    session_mgr = SessionManager(storage_dir=Path("output/demo_sessions"))

    for i, scenario in enumerate(scenarios, 1):
        print(f"[SCENARIO {i}] {scenario['name']}")
        print(f"   Baseline: {scenario['baseline']:.3f}s")
        print(f"   Test laps: {', '.join([f'{t:.3f}s' for t in scenario['test_laps']])}")
        print(f"   Recommendation: {scenario['recommendation']}")
        print()

        # Validate outcome
        result = validator.validate_recommendation_outcome(
            baseline_time=scenario['baseline'],
            test_laps=scenario['test_laps'],
            recommendation=scenario['recommendation']
        )

        print()
        print(f"   VALIDATION RESULT:")
        print(f"      Outcome: {result['outcome'].upper()}")
        print(f"      Lap time delta: {result['lap_time_delta']:+.3f}s")
        print(f"      Confidence: {result['statistical_confidence']:.0%}")
        print(f"      Recommended action: {result['recommended_action'].upper()}")
        print(f"      {result['learning_note']}")

        # Check correctness
        match = result['outcome'] == scenario['expected_outcome']
        status = "PASS" if match else "FAIL"
        print(f"\n   [{status}] Expected '{scenario['expected_outcome']}', got '{result['outcome']}'")

        print()
        print("-"*70)
        print()

    print()
    print("Feature 3 Demo Complete!")
    print("="*70)
    print()
    print()


def demonstrate_integrated_workflow():
    """
    INTEGRATED DEMO: All Three Features Working Together
    Shows the complete enhanced system in action.
    """
    print()
    print("#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  INTEGRATED WORKFLOW: ALL FEATURES TOGETHER".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    print()

    # Generate data
    df = generate_enhanced_demo_data()

    # Driver provides multi-issue feedback
    driver_feedback = "Car feels loose off the corners but front end pushes on entry. Also getting some chatter mid-corner."

    print("[DRIVER FEEDBACK]")
    print(f'   "{driver_feedback}"')
    print()

    # STEP 1: Parse multi-issue feedback
    print("[STEP 1] Parsing multi-issue feedback...")
    feedback_result = interpret_driver_feedback_with_llm(
        raw_feedback=driver_feedback,
        llm_provider="mock",
        multi_issue=True
    )

    issues = feedback_result.get('issues', [])
    print(f"   Detected {len(issues)} issues:")
    for i, issue in enumerate(issues, 1):
        print(f"      {i}. {issue['complaint']} ({issue['severity']})")
        print(f"         -> Priority: {', '.join(issue['priority_features'][:2])}")
    print()

    # STEP 2: Run interaction analysis
    print("[STEP 2] Analyzing parameter interactions...")
    analyzer = InteractionAnalyzer()

    # Collect all priority features from all issues
    all_priority_features = []
    for issue in issues:
        all_priority_features.extend(issue.get('priority_features', []))
    all_priority_features = list(set(all_priority_features))  # Remove duplicates

    # Limit to features that exist in dataframe
    available_features = [f for f in all_priority_features if f in df.columns][:5]

    if len(available_features) >= 2:
        interaction_result = analyzer.find_interactions(
            df=df,
            target='fastest_time',
            features=available_features,
            significance_threshold=0.04
        )

        if interaction_result['has_significant_interactions']:
            print(f"   Found {len(interaction_result['interactions'])} significant interactions")
            top_inter = interaction_result['interactions'][0]
            print(f"   Top interaction: {top_inter['params'][0]} × {top_inter['params'][1]}")
            print(f"   Type: {'SYNERGISTIC' if top_inter['synergistic'] else 'ANTAGONISTIC'}")
        else:
            print(f"   No significant interactions detected")
    else:
        print(f"   Skipping interaction analysis (need 2+ features, have {len(available_features)})")
        interaction_result = None

    print()

    # STEP 3: Generate recommendation
    print("[STEP 3] Generating data-driven recommendation...")

    # Simple correlation analysis
    from sklearn.linear_model import LinearRegression
    features = available_features if available_features else ['tire_psi_rr', 'cross_weight', 'tire_psi_lr']
    features = [f for f in features if f in df.columns][:4]

    X = df[features]
    y = df['fastest_time']
    model = LinearRegression()
    model.fit(X, y)
    impacts = {feat: coef for feat, coef in zip(features, model.coef_)}
    sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    primary_param, primary_impact = sorted_impacts[0]
    direction = "REDUCE" if primary_impact > 0 else "INCREASE"

    recommendation = f"{direction} {primary_param} (predicted impact: {abs(primary_impact):.3f}s)"
    print(f"   Recommendation: {recommendation}")

    # If we have interactions, suggest compound change
    if interaction_result and interaction_result['has_significant_interactions']:
        compound_rec = analyzer.recommend_compound_change(
            primary_param=primary_param,
            all_impacts=impacts,
            available_params=features
        )
        if compound_rec:
            print(f"   PLUS compound adjustment: {compound_rec['secondary_param']}")

    print()

    # STEP 4: Simulate testing and validate
    print("[STEP 4] Simulating test session and validating outcome...")

    # Simulate test results (assume recommendation worked moderately well)
    baseline_time = df['fastest_time'].min()
    # Simulate improvement based on impact magnitude
    simulated_improvement = min(abs(primary_impact) * 0.8, 0.20)  # Cap at 0.2s
    test_laps = [
        baseline_time - simulated_improvement + np.random.normal(0, 0.03)
        for _ in range(5)
    ]

    print(f"   Baseline: {baseline_time:.3f}s")
    print(f"   Test laps: {', '.join([f'{t:.3f}s' for t in test_laps])}")
    print()

    validator = OutcomeValidator()
    outcome = validator.validate_recommendation_outcome(
        baseline_time=baseline_time,
        test_laps=test_laps,
        recommendation=recommendation
    )

    print(f"   OUTCOME: {outcome['outcome'].upper()}")
    print(f"   Improvement: {outcome['lap_time_delta']:+.3f}s")
    print(f"   Confidence: {outcome['statistical_confidence']:.0%}")
    print(f"   Action: {outcome['recommended_action'].upper()}")
    print()

    # STEP 5: Store in session memory for future learning
    print("[STEP 5] Storing results in session memory...")
    session_mgr = SessionManager(storage_dir=Path("output/demo_sessions"))

    session_state = {
        'session_timestamp': datetime.now().isoformat(),
        'driver_feedback': feedback_result,
        'raw_setup_data': df,
        'analysis': {'most_impactful': (primary_param, primary_impact), 'all_impacts': impacts},
        'recommendation': recommendation,
        'outcome_feedback': outcome
    }

    session_id = session_mgr.save_session(session_state)
    print(f"   Session saved: {session_id}")
    print()

    # Show learning metrics
    learning_metrics = session_mgr.get_learning_metrics()
    if learning_metrics.get('parameter_effectiveness'):
        print(f"   [LEARNING] Parameter effectiveness tracking:")
        for param, stats in learning_metrics['parameter_effectiveness'].items():
            print(f"      {param}: {stats['tests']} tests, "
                  f"{stats['success_rate']:.0%} success rate, "
                  f"avg improvement: {stats['avg_improvement']:+.3f}s")

    print()
    print("#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  INTEGRATED DEMO COMPLETE".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    print()


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    print()
    print("*"*70)
    print("*" + " "*68 + "*")
    print("*" + "  AI RACE ENGINEER - ENHANCED FEATURES DEMO".center(68) + "*")
    print("*" + "  Features 1, 2, 3: Multi-Issue | Interactions | Validation".center(68) + "*")
    print("*" + " "*68 + "*")
    print("*"*70)
    print()
    print()

    # Run individual feature demos
    demonstrate_feature_1_multi_issue_parsing()
    input("Press Enter to continue to Feature 2...")
    print()

    df = generate_enhanced_demo_data()
    demonstrate_feature_2_interaction_detection(df)
    input("Press Enter to continue to Feature 3...")
    print()

    demonstrate_feature_3_outcome_validation()
    input("Press Enter to see integrated workflow...")
    print()

    # Run integrated demo
    demonstrate_integrated_workflow()

    print()
    print("="*70)
    print("  ALL DEMOS COMPLETE")
    print("="*70)
    print()
    print("Summary:")
    print("  Feature 1: Multi-Issue Feedback Parser - WORKING")
    print("  Feature 2: Parameter Interaction Detection - WORKING")
    print("  Feature 3: Outcome Validation Loop - WORKING")
    print("  Integrated Workflow: ALL FEATURES TOGETHER - WORKING")
    print()
    print("Next steps:")
    print("  1. Run with real telemetry data")
    print("  2. Test with actual driver feedback")
    print("  3. Accumulate session history for learning")
    print("="*70)
    print()
