# Decision Points Analysis: Where Agents Miss Context

## Overview
This document maps every major decision made by agents to the state information available but unused.

---

## TELEMETRY AGENT - Lines 75-193

### Decision 1: Outlier Detection (Lines 148-174)

**What Agent Decides:**
- Should we remove data points with lap times > threshold?
- Decision made at line 165 or 168 or 172

**Current Logic:**
```python
outlier_threshold = q3 + 1.5 * iqr  # Line 157 - FIXED MULTIPLIER
if len(outliers) < len(df) * 0.2:
    df_clean = df[lap_times <= outlier_threshold]  # Remove
else:
    df_clean = df  # Keep all
```

**Available but Unused Context:**
```python
# FROM STATE
state['session_history']           # Previous session data
state['learning_metrics']          # Historical outlier rates
state['convergence_progress']      # Is system converging?
state['outcome_feedback']          # Did outliers correlate with failed recommendations?

# SPECIFIC MISSED OPPORTUNITIES
# 1. If convergence_progress = 0.8 (80% consistent), outliers might be valuable edge cases
# 2. If previous recommendations improved lap times, old "outliers" are now normal
# 3. If outliers always occur after certain parameter changes, they're not errors
# 4. If model's outcome_feedback shows outliers → better lap times, keep them
```

**Better Decision Would Be:**
```
IF convergence_progress > 0.7 THEN
    # System is converging - be conservative, keep more data
    multiplier = 1.2  (instead of 1.5)
    print("[ADAPTIVE] Keeping potential outliers - system converging")
ELSE IF learning_metrics shows outlier_rate > 25% AND previous outliers worked THEN
    # Outliers are normal in this dataset
    multiplier = 2.0  (be more permissive)
ELSE
    multiplier = 1.5  (standard)
```

**Impact of Missing Context:**
- Too aggressive: Removes good data showing setup improvement
- Too permissive: Includes noise from failed testing sessions
- Not adaptive: Same threshold despite changing setup quality

---

### Decision 2: Sample Size Sufficiency (Lines 176-178)

**What Agent Decides:**
- Do we have enough valid data points to analyze? (Hardcoded: >= 5)

**Current Logic:**
```python
if len(df_clean) < 5:
    return {"error": f"Insufficient data: only {len(df_clean)} valid sessions"}
```

**Available but Unused Context:**
```python
state['convergence_progress']      # If 0.9, maybe 5 is enough; if 0.2, need 15+
state['learning_metrics']          # How many sessions used historically?
state['analysis_strategy']         # Correlation needs 5+, regression needs 10+

# SPECIFIC INSIGHT
# Correlation works with ~5 samples
# Regression needs 10-15 samples
# But we don't know the strategy yet at this point!
```

**Better Decision Would Be:**
```
IF previous sessions < 10 THEN
    required_min = 8  (more conservative early on)
ELSE IF convergence_progress > 0.8 THEN
    required_min = 4  (trust previous learning)
ELSE
    required_min = 5  (default)

if len(df_clean) < required_min:
    return error
```

**Impact of Missing Context:**
- Might reject borderline datasets that would work
- Doesn't account for skill of driver/test variation over time

---

## ANALYSIS AGENT - Lines 196-355

### Decision 3: Feature Selection (Lines 214-257)

**What Agent Decides:**
- Which setup parameters should we analyze? (Lines 230-247)

**Current Logic:**
```python
for feature in potential_features:
    variance = df[feature].std()
    if variance > 0.01:  # ONLY criterion
        selected_features.append(feature)
```

**Available but Unused Context:**
```python
state['learning_metrics']['parameter_impacts']  # Historical correlation for each param
state['previous_recommendations']               # What parameters were recommended before?
state['outcome_feedback']                       # Which params actually worked?

# EXAMPLE OF MISSED INSIGHT
# Parameter A: variance = 0.5, correlation = -0.02  (high variance, useless)
# Parameter B: variance = 0.1, correlation = -0.45  (low variance, crucial)
# Currently: Both included if variance > 0.01, but wrong one is excluded if variance < 0.01
```

**Better Decision Would Be:**
```
for feature in potential_features:
    variance = df[feature].std()
    
    # Check if parameter worked historically
    param_history = learning_metrics.get('parameter_impacts', {}).get(feature, [])
    
    # Include if:
    # 1. High variance AND any correlation history (might be important)
    # 2. Low variance BUT has strong historical correlation (crucial parameter)
    # 3. New parameter (no history) AND has variance
    # Exclude if:
    # 1. High variance but consistent failure history
    # 2. Low variance AND low correlation history AND not in priority_features
    
    if param_history:
        avg_correlation = np.mean([abs(c) for c in param_history])
        if variance > 0.01 or avg_correlation > 0.15:
            selected_features.append(feature)
    elif variance > 0.01:
        selected_features.append(feature)
```

**Impact of Missing Context:**
- Includes parameters with no predictive power
- Excludes subtle-but-important parameters
- Wastes analysis budget on variable-but-useless parameters

---

### Decision 4: Strategy Selection (Lines 258-282)

**What Agent Decides:**
- Use correlation or regression analysis? (Line 282)

**Current Logic:**
```python
if sample_size < 10:
    strategy = "correlation"      # Assumes good for small N
elif feature_count > sample_size / 2:
    strategy = "correlation"      # Assumes good for high-dim
elif variance < 0.15:
    strategy = "correlation"      # Assumes good for low variance
else:
    strategy = "regression"
```

**Available but Unused Context:**
```python
state['learning_metrics']['strategy_performance']  # Historical success of each strategy
# {
#     'correlation': {'runs': 12, 'successful': 10, 'success_rate': 0.83},
#     'regression': {'runs': 8, 'successful': 3, 'success_rate': 0.38}
# }

state['convergence_progress']  # 0.8 convergence means we know what we're doing

# SPECIFIC EXAMPLE
# If correlation succeeded 83% of the time and regression only 38%,
# we should use correlation regardless of the current dataset characteristics
```

**Better Decision Would Be:**
```
if strategy_performance:
    corr_success = strategy_performance['correlation']['success_rate']
    reg_success = strategy_performance['regression']['success_rate']
    
    if corr_success > reg_success and corr_success > 0.6:
        strategy = 'correlation'
        reason = f"historical: {corr_success:.0%} success rate"
    elif reg_success > 0.6:
        strategy = 'regression'
        reason = f"historical: {reg_success:.0%} success rate"
    else:
        # Use rule-based approach as fallback
        strategy = 'correlation' if sample_size < 10 else 'regression'
else:
    # First run, use rule-based
    ... [current logic] ...
```

**Impact of Missing Context:**
- Chooses wrong model type for data characteristics
- Doesn't learn from past failures
- Same decision rule every time despite changing data patterns

---

## ENGINEER AGENT - Lines 358-551

### Decision 5: Parameter vs Driver Feedback (Lines 383-424)

**What Agent Decides:**
- Trust data recommendation or driver diagnosis? (Lines 390-423)

**Current Logic:**
```python
if param in priority_features:
    print("VALIDATION: Top parameter matches driver feedback!")
    decision_rationale = "driver_validated_by_data"
else:
    # Conflict - try to use driver's priority list instead
    best_priority_param = max(priority_impacts.items(), ...)
    decision_rationale = "driver_feedback_prioritized"
```

**Available but Unused Context:**
```python
state['previous_recommendations']  # What's the pattern of recent choices?
state['outcome_feedback']          # Did previous driver-based recs work?
state['learning_metrics']['parameter_outcomes']  # Success rate of data vs driver params

# EXAMPLE OF MISSED INSIGHT
# Previous 5 recommendations:
# 1. Data-based: tire_psi_lf → improved ✓
# 2. Data-based: cross_weight → no change ✗
# 3. Driver-based: spring_rf → improved ✓
# 4. Data-based: tire_psi_rf → worse ✗
# 5. Driver-based: track_bar → improved ✓
#
# Pattern: Data correct 40%, Driver correct 67% → TRUST DRIVER
```

**Better Decision Would Be:**
```
# Score which approach works better
data_success_rate = calculate_success_rate(learning_metrics, method='data_based')
driver_success_rate = calculate_success_rate(learning_metrics, method='driver_based')

if data_success_rate > driver_success_rate + 0.2:
    # Data consistently better by 20%+
    print("TRUST DATA: Better historical performance")
    decision_rationale = "data_prioritized_justified"
elif driver_success_rate > data_success_rate + 0.2:
    # Driver consistently better
    print("TRUST DRIVER: Better historical performance")
    decision_rationale = "driver_prioritized_justified"
else:
    # Roughly equal - combine both
    # Recommend driver param but note data alternative
    decision_rationale = "hybrid_approach"
```

**Impact of Missing Context:**
- Repeats mistake of trusting source that historically failed
- Doesn't learn driver is actually better at certain car characteristics
- Doesn't learn data is better at certain parameters

---

### Decision 6: Signal Strength Assessment (Lines 429-438)

**What Agent Decides:**
- Is impact value STRONG, MODERATE, or WEAK? (Lines 431-438)

**Current Logic:**
```python
if abs(impact) > 0.1:
    signal_strength = "STRONG"        # Hardcoded threshold
elif abs(impact) > 0.05:
    signal_strength = "MODERATE"      # Hardcoded threshold
else:
    signal_strength = "WEAK"
```

**Available but Unused Context:**
```python
state['learning_metrics']['parameter_impacts'][param]  # Historical impacts for this param
# Example: [-0.12, -0.09, -0.11, -0.08]  => CONSISTENT despite being "weak"

state['convergence_progress']  # 0.9 means we're locked in, signals should be trusted
state['previous_recommendations']  # How many times recommended this param?

# SPECIFIC EXAMPLE
# This session: tire_psi_rf impact = 0.08 (WEAK threshold < 0.1)
# Historical: [0.09, 0.08, 0.07, 0.09] (VERY CONSISTENT)
# Convergence: 0.8 (system is confident)
# => Should be: WEAK_BUT_VERY_RELIABLE (higher confidence than new "strong" signal)
```

**Better Decision Would Be:**
```
param_history = learning_metrics.get('parameter_impacts', {}).get(param, [])

if param_history and len(param_history) >= 2:
    # Score consistency: low variance = consistent
    impact_variance = np.std(param_history)
    impact_mean = np.mean(param_history)
    consistency = 1 - (impact_variance / (abs(impact_mean) + 0.001))
else:
    consistency = 0.5

# Adjust thresholds based on consistency
if consistency > 0.8:
    # Very consistent - lower thresholds
    threshold_strong = 0.07
    threshold_moderate = 0.03
elif consistency > 0.5:
    # Normal - standard thresholds
    threshold_strong = 0.1
    threshold_moderate = 0.05
else:
    # Inconsistent - raise thresholds (need stronger signal)
    threshold_strong = 0.15
    threshold_moderate = 0.08

if abs(impact) > threshold_strong:
    signal_strength = "STRONG"
elif abs(impact) > threshold_moderate:
    signal_strength = "MODERATE"
else:
    signal_strength = "WEAK"

# Add consistency info
print(f"Signal strength: {signal_strength} (consistency: {consistency:.0%})")
```

**Impact of Missing Context:**
- Gives false confidence to one-off strong signals
- Ignores consistent weak signals that are actually reliable
- Same thresholds for data with vastly different reliability

---

### Decision 7: Recommendation Generation (Lines 440-483)

**What Agent Decides:**
- What to recommend? (Lines 441-483)

**Current Logic:**
```python
if signal_strength == "STRONG":
    rec = f"STRONG RECOMMENDATION: {direction} {param}..."
elif signal_strength == "MODERATE":
    rec = f"MODERATE RECOMMENDATION: Consider {direction} {param}..."
else:  # WEAK
    rec = f"NO DOMINANT PARAMETER FOUND\n" + \
          f"Recommendation: Test interaction effects between..."
```

**Available but Unused Context:**
```python
state['learning_metrics']['parameter_outcomes'][param]
# {
#     'test_count': 6,
#     'successful': 1,          # Only worked 1 time!
#     'success_rate': 0.17,
#     'last_outcome': 'worse'
# }

state['previous_recommendations']  # Last 3 were all same parameter - redundant?
state['session_history']  # Recent sessions show diverging results

# CRITICAL INSIGHT NOT USED
# Parameter might have "STRONG" impact mathematically
# But failed in 83% of actual implementations
# => Should reduce confidence despite strong signal
```

**Better Decision Would Be:**
```
param_outcomes = learning_metrics.get('parameter_outcomes', {}).get(param, {})
success_rate = param_outcomes.get('success_rate', 0.5)
test_count = param_outcomes.get('test_count', 0)

# Adjust recommendation confidence based on actual outcomes
if test_count >= 3:  # Enough historical data
    if success_rate < 0.3:
        # Recommendation failed often - warn user
        rec_header = f"[CAUTION] {param} has only {success_rate:.0%} success rate\n"
        rec_header += "Consider alternative parameters:\n"
        # List alternatives
        signal_strength = "WEAK_UNRELIABLE"
    elif success_rate < 0.6:
        # Mixed results - moderate confidence
        signal_strength = "INCONSISTENT"
    # else: >= 0.6 success, trust the recommendation

# Check for repetition
if previous_recommendations:
    recent_params = [param for _, param in previous_recommendations[-3:]]
    if recent_params.count(param) >= 2:
        rec += "\n[NOTE] This parameter was recommended 2+ times recently."
        rec += "\nConsider testing alternatives if still not improved."

# Generate final recommendation
if signal_strength == "WEAK_UNRELIABLE":
    rec = f"ALTERNATIVE RECOMMENDATION\n" + \
          f"   Previous attempts at {param} had low success.\n" + \
          f"   Try: {next_best_param} instead"
elif signal_strength == "INCONSISTENT":
    rec = f"MIXED SIGNAL: {param} sometimes helps\n" + \
          f"   Success rate: {success_rate:.0%}\n" + \
          f"   Recommendation: Test {param} again carefully, monitor closely"
else:
    # Standard recommendation as before
    ...
```

**Impact of Missing Context:**
- Recommends parameters that failed consistently
- Doesn't warn about low success rates
- Doesn't rotate between alternatives
- Keeps recommending same parameter despite pattern of failures

---

### Decision 8: Multi-Parameter Interaction Testing (Lines 473-483)

**What Agent Decides:**
- When signal is weak, recommend interaction testing?

**Current Logic:**
```python
else:  # WEAK signal
    top_3 = sorted(all_impacts.items(), ...)[:3]
    rec = f"Recommendation: Test interaction effects between:\n" + \
          f"      • " + "\n      • ".join(params)
    print("Multi-parameter interaction testing recommended")
```

**Available but Unused Context:**
```python
# NO ACTUAL INTERACTION ANALYSIS DONE
# Just list parameters - doesn't calculate which pairs interact
# Missing: state['learning_metrics']['interactions']

# If we had interaction history:
# {
#     'tire_psi_rr × spring_rr': {'correlation': -0.42, 'tested': 3},
#     'cross_weight × tire_psi_lf': {'correlation': 0.05, 'tested': 2}
# }

# INSIGHT NOT USED
# tire_psi_rr × spring_rr work well together (tested 3x)
# cross_weight × tire_psi_lf don't interact (weak correlation)
```

**Better Decision Would Be:**
```python
else:  # WEAK signal
    # Find synergistic parameter pairs from history
    interactions = learning_metrics.get('interactions', {})
    
    synergistic_pairs = [
        (pair, data) for pair, data in interactions.items()
        if abs(data.get('correlation', 0)) > 0.1 and data.get('tested', 0) >= 2
    ]
    
    if synergistic_pairs:
        # Found parameters that work well together
        best_pair = max(synergistic_pairs, key=lambda x: x[1]['correlation'])
        param1, param2 = best_pair[0].split(' × ')
        
        rec = f"SYNERGISTIC PARAMETERS IDENTIFIED\n"
        rec += f"   Adjust BOTH {param1} and {param2} together\n"
        rec += f"   Interaction strength: {best_pair[1]['correlation']:.2f}\n"
        rec += f"   (Tested successfully {best_pair[1]['tested']} times)"
        
        print("DECISION: Multi-parameter synergistic adjustment")
    else:
        # No strong interaction history - need more data
        rec = f"INSUFFICIENT DATA\n" + \
              f"   Current data doesn't show dominant single parameter.\n" + \
              f"   Consider:\n" + \
              f"      1. Collect more test sessions\n" + \
              f"      2. Focus on parameters: {', '.join(top_3)}\n" + \
              f"      3. Try adjusting multiple parameters simultaneously"
        
        print("DECISION: Need more data or structured multi-param testing")
```

**Impact of Missing Context:**
- Recommends interaction testing but doesn't identify which pairs
- Doesn't leverage historical interaction knowledge
- Wastes testing time on non-interactive parameters

---

## SUMMARY: Missing Context by Type

### Unused State Fields

| Field | Used Where | Missing From | Cost of Not Using |
|-------|-----------|--------------|-------------------|
| `session_history` | LLM only (line 524) | All decisions | Can't validate patterns |
| `learning_metrics` | Display only (demo line 114) | All agent decisions | No statistical memory |
| `previous_recommendations` | Never | Engineer agent decisions | Repeats same failures |
| `outcome_feedback` | Saved only (line 253) | All validation | No feedback loop |
| `convergence_progress` | Never | Confidence thresholds | Wrong decision confidence |
| `parameter_impacts` (in metrics) | Never | Feature selection | Includes useless parameters |
| `strategy_performance` (in metrics) | Never | Strategy selection | Doesn't learn from failures |
| `parameter_outcomes` (in metrics) | Never | Parameter recommendations | Recommends failing parameters |

### Decision Improvements Needed

| Agent | Line | Decision | Improvement | Impact |
|-------|------|----------|-------------|--------|
| Telemetry | 157 | Outlier threshold | Adaptive based on convergence | Better data quality |
| Telemetry | 177 | Sample size | Adaptive based on strategy/convergence | Right amount of data |
| Analysis | 238 | Feature selection | Use correlation history, exclude failures | Better feature set |
| Analysis | 269 | Strategy selection | Use historical success rates | Right model choice |
| Engineer | 390 | Data vs Driver | Compare historical success rates | Better decision source |
| Engineer | 430 | Signal strength | Adjust thresholds by consistency | Reliable confidence |
| Engineer | 376 | Parameter selection | Check success rate history | Avoid failed params |
| Engineer | 480 | Interaction testing | Actually calculate interactions | Find synergies |

