# AI Race Engineer - Improvement Roadmap

## Phase 1: State Management Integration (High Impact, Low Effort)

### 1.1 Telemetry Agent - Use Learning Metrics for Outlier Detection
**File:** race_engineer.py, lines 148-174
**Effort:** 2-3 hours
**Impact:** Prevents discarding good data points, adapts to improving setup

```python
# CURRENT (lines 154-158)
q1, q3 = lap_times.quantile(0.25), lap_times.quantile(0.75)
iqr = q3 - q1
outlier_threshold = q3 + 1.5 * iqr
outliers = df[lap_times > outlier_threshold]

# IMPROVED VERSION
learning_metrics = state.get('learning_metrics', {})
historic_outlier_rate = learning_metrics.get('outlier_removal_rate', 0.15)

# Adapt threshold based on whether system is converging
convergence = state.get('convergence_progress', 0)
if convergence > 0.7:  # System converging - be less aggressive
    iqr_multiplier = 1.2  # Keep more data
else:
    iqr_multiplier = 1.5  # Standard outlier detection

q1, q3 = lap_times.quantile(0.25), lap_times.quantile(0.75)
iqr = q3 - q1
outlier_threshold = q3 + iqr_multiplier * iqr
outliers = df[lap_times > outlier_threshold]

print(f"   [ADAPTIVE] Outlier threshold: {outlier_threshold:.3f}s")
print(f"      Convergence-adjusted multiplier: {iqr_multiplier}")
print(f"      Historical outlier rate: {historic_outlier_rate:.1%}")
```

### 1.2 Analysis Agent - Adaptive Feature Selection
**File:** race_engineer.py, lines 214-257
**Effort:** 2-3 hours
**Impact:** Focuses on parameters that worked before, excludes consistently-failing parameters

```python
# CURRENT (lines 230-247)
for feature in potential_features:
    if feature not in df.columns:
        continue
    variance = df[feature].std()
    if variance > 0.01:
        selected_features.append(feature)

# IMPROVED VERSION
learning_metrics = state.get('learning_metrics', {})
parameter_outcomes = learning_metrics.get('parameter_outcomes', {})  # NEW in session_manager

for feature in potential_features:
    if feature not in df.columns:
        continue
    variance = df[feature].std()
    
    # Check outcome history
    feature_outcomes = parameter_outcomes.get(feature, {})
    success_rate = feature_outcomes.get('success_rate', 0.5)  # Default: neutral
    test_count = feature_outcomes.get('test_count', 0)
    
    # Feature must have variance AND good historical performance
    variance_ok = variance > 0.01
    outcome_ok = success_rate > 0.4 or test_count < 3  # Give new params a chance
    
    if variance_ok and outcome_ok:
        selected_features.append(feature)
        
        # Mark in output if this parameter failed before
        if test_count >= 3 and success_rate < 0.3:
            print(f"   [WARNING] {feature}: High failure rate ({success_rate:.0%}), reconsidering...")
```

### 1.3 Engineer Agent - Parameter Effectiveness Checking
**File:** race_engineer.py, lines 376-426
**Effort:** 3-4 hours
**Impact:** Stops recommending parameters that failed in past sessions

```python
# ADD THIS FUNCTION
def _get_parameter_effectiveness(
    param: str,
    learning_metrics: Optional[Dict]
) -> Dict:
    """
    Score parameter effectiveness from historical outcomes.
    
    Returns:
        {
            'success_rate': 0.66,      # 2 successes out of 3 tests
            'test_count': 3,
            'avg_improvement': 0.032,  # seconds
            'last_outcome': 'improved', # or 'no_change', 'worse'
            'reliability': 'RELIABLE' | 'INCONSISTENT' | 'UNTESTED'
        }
    """
    if not learning_metrics:
        return {'reliability': 'UNTESTED'}
    
    outcomes = learning_metrics.get('parameter_outcomes', {})
    param_history = outcomes.get(param, {})
    
    test_count = param_history.get('test_count', 0)
    successful = param_history.get('successful', 0)
    
    if test_count == 0:
        return {'reliability': 'UNTESTED'}
    
    success_rate = successful / test_count
    
    # Reliability scoring
    if test_count < 2:
        reliability = 'UNTESTED'
    elif success_rate < 0.3:
        reliability = 'UNRELIABLE'
    elif success_rate < 0.7:
        reliability = 'INCONSISTENT'
    else:
        reliability = 'RELIABLE'
    
    return {
        'success_rate': success_rate,
        'test_count': test_count,
        'avg_improvement': param_history.get('avg_improvement', 0),
        'last_outcome': param_history.get('last_outcome'),
        'reliability': reliability
    }

# INTEGRATE IN engineer_agent (line 376-426)
# Add after line 381
effectiveness = _get_parameter_effectiveness(param, learning_metrics)
print(f"   [HISTORY] Parameter effectiveness: {effectiveness['reliability']}")
if effectiveness['reliability'] == 'UNRELIABLE':
    print(f"      WARNING: This parameter failed {1-effectiveness['success_rate']:.0%} of the time")
    print(f"      Consider alternative parameters instead")
```

---

## Phase 2: Analysis Enhancement (Medium Impact, Medium Effort)

### 2.1 Implement Interaction Term Analysis
**File:** race_engineer.py, add new method, update engineer_agent (lines 473-483)
**Effort:** 4-5 hours
**Impact:** Finds multi-parameter optimization opportunities

```python
def _analyze_interactions(df: pd.DataFrame, top_features: List[str], target: str = 'fastest_time') -> Dict:
    """
    Calculate interaction effects between top parameters.
    
    Returns:
        {
            'interactions': [
                {'param1': 'tire_psi_rr', 'param2': 'spring_rr', 'correlation': -0.42},
                ...
            ],
            'synergistic': [],  # Parameters that work well together
            'conflicting': []   # Parameters that work against each other
        }
    """
    interactions = []
    
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            # Create interaction term
            interaction = df[feat1] * df[feat2]
            corr = interaction.corr(df[target])
            
            if abs(corr) > 0.05:  # Significant interaction
                interactions.append({
                    'param1': feat1,
                    'param2': feat2,
                    'correlation': float(corr),
                    'type': 'synergistic' if corr < -0.1 else 'conflicting'
                })
    
    # Separate synergistic from conflicting
    synergistic = [i for i in interactions if i['type'] == 'synergistic']
    conflicting = [i for i in interactions if i['type'] == 'conflicting']
    
    return {
        'interactions': sorted(interactions, key=lambda x: abs(x['correlation']), reverse=True),
        'synergistic': synergistic,
        'conflicting': conflicting
    }

# UPDATE engineer_agent, lines 473-483
else:  # WEAK signal
    # IMPROVED: Actually compute interactions
    top_3 = [p for p, _ in sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]]
    interactions = _analyze_interactions(df, top_3)
    
    if interactions['synergistic']:
        # Found parameters that work together
        rec = f"SYNERGISTIC PARAMETERS IDENTIFIED\n" + \
              f"   These parameters work well together:\n"
        for inter in interactions['synergistic'][:2]:
            rec += f"      • {inter['param1']} + {inter['param2']} (interaction: {inter['correlation']:.3f})\n"
        rec += f"   Recommendation: Adjust BOTH parameters together\n"
        print(f"    DECISION: Multi-parameter combined recommendation")
    else:
        rec = f"NO DOMINANT PARAMETER FOUND\n" + \
              f"   Consider testing interaction effects between: " + \
              f"{', '.join(top_3)}\n" + \
              f"   Collect more data (current: {len(df)} sessions)"
        print(f"    DECISION: Need more data for interaction analysis")
```

### 2.2 Adaptive Signal Strength Thresholds
**File:** race_engineer.py, lines 429-438
**Effort:** 2 hours
**Impact:** Prevents false confidence in weak parameters

```python
# CURRENT (lines 430-438)
if abs(impact) > 0.1:
    signal_strength = "STRONG"
elif abs(impact) > 0.05:
    signal_strength = "MODERATE"
else:
    signal_strength = "WEAK"

# IMPROVED VERSION
learning_metrics = state.get('learning_metrics', {})
parameter_impacts = learning_metrics.get('parameter_impacts', {})
param_history = parameter_impacts.get(param, [])

# Calculate consistency of this parameter's impact
if param_history and len(param_history) >= 2:
    impact_variance = np.std(param_history)
    impact_consistency = 1 - min(impact_variance / (abs(np.mean(param_history)) + 0.01), 1)
else:
    impact_consistency = 0.5  # Unknown consistency

# Adjust thresholds based on consistency
if impact_consistency > 0.8:  # Very consistent
    threshold_strong = 0.08
    threshold_moderate = 0.04
elif impact_consistency > 0.5:  # Somewhat consistent
    threshold_strong = 0.10
    threshold_moderate = 0.05
else:  # Inconsistent
    threshold_strong = 0.15
    threshold_moderate = 0.08

if abs(impact) > threshold_strong:
    signal_strength = "STRONG"
    print(f"    DECISION: Strong signal (consistency: {impact_consistency:.0%})")
elif abs(impact) > threshold_moderate:
    signal_strength = "MODERATE"
    print(f"    DECISION: Moderate signal (consistency: {impact_consistency:.0%})")
else:
    signal_strength = "WEAK"
    print(f"    DECISION: Weak signal (consistency: {impact_consistency:.0%})")
```

---

## Phase 3: Feedback Loop Implementation (High Impact, High Effort)

### 3.1 Add Outcome Feedback Collection to Demo
**File:** demo.py, add new section after line 225
**Effort:** 3-4 hours
**Impact:** Enables learning from recommendation outcomes

```python
# ADD AFTER line 225 (session saved)
print("\n[5/6] Outcome Validation...")
print()

# In production, this would be automated by comparing next session's data
# For now, offer manual feedback
print("Did the recommendation improve lap times?")
print("  Options: 'yes', 'no', 'skip'")
print("  Example: 'yes' (if lap time improved after applying change)")
print()

outcome_choice = input("Outcome: ").strip().lower()

if outcome_choice == 'yes':
    improvement_str = input("  Lap time improvement (seconds): ").strip()
    try:
        improvement = float(improvement_str)
        outcome_feedback = {
            'driver_assessment': 'improved',
            'lap_time_improvement': improvement,
            'validated': True,
            'timestamp': datetime.now().isoformat()
        }
        session_mgr.add_outcome_feedback(session_id, outcome_feedback)
        print(f"  [OK] Recorded improvement: {improvement:+.3f}s")
    except ValueError:
        print("  [WARNING] Invalid improvement value, skipping")
        
elif outcome_choice == 'no':
    outcome_feedback = {
        'driver_assessment': 'no_change',
        'lap_time_improvement': 0,
        'validated': True,
        'timestamp': datetime.now().isoformat()
    }
    session_mgr.add_outcome_feedback(session_id, outcome_feedback)
    print(f"  [OK] Recorded: recommendation had no effect")
```

### 3.2 Update SessionManager - Calculate Parameter Effectiveness
**File:** session_manager.py, add new method after line 201
**Effort:** 4-5 hours
**Impact:** Provides effectiveness scores for agent decision-making

```python
def calculate_parameter_effectiveness(self) -> Dict:
    """
    Analyze parameter effectiveness from outcome history.
    
    Returns:
        {
            'parameter_outcomes': {
                'tire_psi_lf': {
                    'test_count': 5,
                    'successful': 3,
                    'success_rate': 0.6,
                    'avg_improvement': 0.025,
                    'last_outcome': 'improved'
                },
                ...
            }
        }
    """
    sessions = self.load_session_history(limit=50)  # Get more history
    
    parameter_outcomes = {}
    
    for session in sessions:
        param = self._extract_parameter_from_recommendation(session.get('recommendation', ''))
        if not param:
            continue
        
        if param not in parameter_outcomes:
            parameter_outcomes[param] = {
                'test_count': 0,
                'successful': 0,
                'failed': 0,
                'improvements': [],
                'last_outcome': None
            }
        
        parameter_outcomes[param]['test_count'] += 1
        
        # Check outcome feedback
        outcome = session.get('outcome_feedback')
        if outcome:
            if outcome.get('driver_assessment') == 'improved':
                parameter_outcomes[param]['successful'] += 1
                improvement = outcome.get('lap_time_improvement', 0)
                parameter_outcomes[param]['improvements'].append(improvement)
            else:
                parameter_outcomes[param]['failed'] += 1
            
            parameter_outcomes[param]['last_outcome'] = outcome.get('driver_assessment')
    
    # Calculate derived metrics
    for param, data in parameter_outcomes.items():
        if data['test_count'] > 0:
            data['success_rate'] = data['successful'] / data['test_count']
            if data['improvements']:
                data['avg_improvement'] = np.mean(data['improvements'])
            else:
                data['avg_improvement'] = 0
    
    return {'parameter_outcomes': parameter_outcomes}

# INTEGRATE with _update_learning_metrics (line 137)
# Add this call at the end of _update_learning_metrics:
effectiveness = self.calculate_parameter_effectiveness()
metrics['parameter_outcomes'] = effectiveness['parameter_outcomes']
```

---

## Phase 4: Strategy Learning (Medium Impact, Medium Effort)

### 4.1 Track Strategy Performance
**File:** session_manager.py, add tracking of which strategy was used
**Effort:** 2-3 hours
**Impact:** Allows agents to choose strategies based on historical performance

```python
# IN _update_learning_metrics (line 137), add:
strategy = session_data.get('analysis_strategy', 'unknown')
r_squared = session_data.get('analysis', {}).get('r_squared', None)

if 'strategy_performance' not in metrics:
    metrics['strategy_performance'] = {
        'correlation': {'runs': 0, 'successful': 0, 'avg_r_squared': 0},
        'regression': {'runs': 0, 'successful': 0, 'avg_r_squared': 0}
    }

if strategy in metrics['strategy_performance']:
    perf = metrics['strategy_performance'][strategy]
    perf['runs'] += 1
    
    # Check if recommendation was validated
    outcome = session_data.get('outcome_feedback')
    if outcome and outcome.get('driver_assessment') == 'improved':
        perf['successful'] += 1
    
    # Track R² for regression
    if r_squared is not None:
        perf['avg_r_squared'] = (perf['avg_r_squared'] * (perf['runs'] - 1) + r_squared) / perf['runs']

# Calculate success rates
for strategy in ['correlation', 'regression']:
    perf = metrics['strategy_performance'][strategy]
    if perf['runs'] > 0:
        perf['success_rate'] = perf['successful'] / perf['runs']
```

### 4.2 Use Strategy Performance in Analysis Agent
**File:** race_engineer.py, lines 268-282
**Effort:** 2 hours
**Impact:** Chooses better-performing strategy for similar data

```python
# REPLACE lines 268-282 with:
sample_size = len(df)
feature_count = len(selected_features)
variance = df['fastest_time'].std()

print(f"\n   [ANALYSIS] Evaluating strategy options...")
print(f"      • Sample size: {sample_size}")
print(f"      • Feature count: {feature_count}")
print(f"      • Lap time variance: {variance:.3f}s")

# Check historical performance
learning_metrics = state.get('learning_metrics', {})
strategy_performance = learning_metrics.get('strategy_performance', {})

strategy = "correlation"
reason = "default"

if strategy_performance:
    corr_success = strategy_performance.get('correlation', {}).get('success_rate', 0)
    reg_success = strategy_performance.get('regression', {}).get('success_rate', 0)
    
    if reg_success > corr_success and reg_success > 0.6:
        strategy = "regression"
        reason = f"historical: regression {reg_success:.0%} > correlation {corr_success:.0%}"
    elif corr_success > 0.6:
        strategy = "correlation"
        reason = f"historical: correlation {corr_success:.0%} performs better"
    elif sample_size >= 15 and feature_count <= sample_size / 3:
        strategy = "regression"
        reason = "adequate data quality"
    else:
        strategy = "correlation"
        reason = "conservative approach for limited data"

print(f"    DECISION: Using {strategy.upper()} analysis ({reason})")
```

---

## Implementation Priority

### Quick Wins (< 2 hours each)
1. ✓ Use convergence_progress in outlier detection (1.1)
2. ✓ Adaptive signal strength thresholds (2.2)
3. ✓ Parameter effectiveness checking function (1.3)

### High Value (2-4 hours)
1. Feature selection with outcome history (1.2)
2. Interaction term analysis implementation (2.1)
3. Outcome feedback collection (3.1)

### Medium Effort, High Impact (4+ hours)
1. SessionManager effectiveness calculation (3.2)
2. Strategy performance tracking (4.1)
3. Use strategy performance in decisions (4.2)

---

## Testing Strategy

For each improvement:
1. **Unit test** the new function with mock data
2. **Integration test** in demo.py with real/mock data
3. **Regression test** to ensure existing functionality works
4. **Output validation** to see improved decision explanations

Example:
```python
# Test parameter effectiveness checking
learning_metrics_mock = {
    'parameter_outcomes': {
        'tire_psi_lf': {
            'test_count': 5,
            'successful': 1,
            'avg_improvement': 0.002,
            'last_outcome': 'worse'
        }
    }
}

effectiveness = _get_parameter_effectiveness('tire_psi_lf', learning_metrics_mock)
assert effectiveness['reliability'] == 'UNRELIABLE'
assert effectiveness['success_rate'] == 0.2
print("[PASS] Parameter effectiveness detection works")
```

