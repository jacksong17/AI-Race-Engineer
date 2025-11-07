# Data Structures and State Evolution Reference

## Complete State Dictionary Lifecycle

### 1. Initial State (demo.py, line 152-161)

```python
initial_state = {
    # ===== INPUT DATA =====
    'raw_setup_data': DataFrame(
        # Session-level data with setup parameters
        columns=['session_id', 'fastest_time', 'tire_psi_lf', 'tire_psi_rf', 
                 'tire_psi_lr', 'tire_psi_rr', 'cross_weight', 'track_bar_height_left', 
                 'spring_lf', 'spring_rf', ...]
        # Example row: {session_id: 'bristol_test_1', fastest_time: 15.213, 
        #              tire_psi_lf: 28.0, tire_psi_rf: 32.0, ...}
    ),
    
    'driver_feedback': {
        'complaint': 'loose_exit',                    # Classification
        'severity': 'moderate',                       # Level
        'phase': 'corner_exit',                       # When it occurs
        'diagnosis': 'Insufficient rear grip...',     # Technical assessment
        'priority_features': ['tire_psi_rr', 'tire_psi_lr', 'track_bar_height_left']
    },
    
    # ===== INITIAL PROCESSING =====
    'data_quality_decision': None,                    # Will be set by Agent 1
    'driver_diagnosis': None,                         # Will be set by Agent 1
    'analysis_strategy': None,                        # Will be set by Agent 2
    'selected_features': None,                        # Will be set by Agent 2
    'analysis': None,                                 # Will be set by Agent 2
    'recommendation': None,                           # Will be set by Agent 3
    'error': None                                     # Error tracking
}
```

---

### 2. After Agent 1: Telemetry Chief (race_engineer.py, lines 67-185)

**State additions from telemetry_agent():**

```python
{
    # ===== UNCHANGED =====
    'raw_setup_data': DataFrame(...),  # Original preserved
    'driver_feedback': {...},
    
    # ===== AGENT 1 OUTPUT =====
    'driver_diagnosis': {
        'diagnosis': 'Oversteer (loose rear end)',
        'technical_cause': 'Insufficient rear grip - likely rear tire pressure...',
        'priority_features': ['tire_psi_rr', 'tire_psi_lr', 'track_bar_height_left', 
                             'spring_rf', 'spring_rr'],
        'complaint_type': 'loose_exit'
    },
    
    'data_quality_decision': 'removed_3_outliers',  # Options:
                                                    # - 'no_outliers_found'
                                                    # - 'kept_all_data'
                                                    # - 'removed_N_outliers'
    
    'raw_setup_data': DataFrame(...)  # UPDATED: cleaned, outliers removed
                                      # Same structure but fewer rows
    
    # ===== UNCHANGED =====
    'analysis_strategy': None,
    'selected_features': None,
    'analysis': None,
    'recommendation': None,
    'error': None
}
```

**Decision Flow in Agent 1:**
```
Driver Feedback "Car feels loose coming off the corners"
    ├─ Contains: "loose" → Oversteer detection
    ├─ Contains: "corners" → Phase = "mid_corner" or "corner_exit"
    └─ Maps to priority_features for oversteer
        ├─ tire_psi_rr (rear right pressure)
        ├─ tire_psi_lr (rear left pressure)
        ├─ track_bar_height_left
        ├─ spring_rf, spring_rr
        └─ These will be PRIORITIZED in Agent 2's analysis

Data Quality Assessment
    ├─ IQR outlier detection on fastest_time
    │   ├─ Q1, Q3 calculated
    │   ├─ IQR = Q3 - Q1
    │   └─ Threshold = Q3 + 1.5 * IQR
    │
    └─ Decision: Remove < 20% outliers, keep all if >= 20%
        └─ Reasoning: Too many outliers might indicate systematic issue
```

---

### 3. After Agent 2: Data Scientist (race_engineer.py, lines 188-347)

**State additions from analysis_agent():**

```python
{
    # ===== PRESERVED FROM PREVIOUS =====
    'raw_setup_data': DataFrame(...),  # Cleaned by Agent 1
    'driver_feedback': {...},
    'driver_diagnosis': {...},
    'data_quality_decision': 'removed_3_outliers',
    
    # ===== AGENT 2 OUTPUT =====
    'selected_features': [
        'tire_psi_lf', 'tire_psi_rf', 'tire_psi_lr', 'tire_psi_rr',
        'cross_weight', 'track_bar_height_left', 'spring_lf', 'spring_rf'
    ],
    # These are features with variance > 0.01 (were actually changed in testing)
    # Excluded: parameters that stayed constant across all sessions
    
    'analysis_strategy': 'regression',  # Options: 'correlation' or 'regression'
    # Decision based on:
    # - Sample size: >= 10 good for regression
    # - Feature count: avoid if > sample_size/2
    # - Variance: < 0.15 suggests correlation better
    
    'analysis': {
        'method': 'regression',  # What method was actually used
        
        'all_impacts': {
            'tire_psi_rr': 0.551,      # Positive = increase makes lap time worse (slower)
            'tire_psi_lf': -0.289,     # Negative = increase makes lap time better (faster)
            'cross_weight': -0.195,
            'track_bar_height_left': 0.112,
            'tire_psi_rf': 0.087,
            'tire_psi_lr': -0.043,
            'spring_rf': -0.031,
            'spring_lf': 0.021
        },
        # Interpretation:
        # - tire_psi_rr: Most impactful. REDUCE it for 0.551s improvement per unit
        # - tire_psi_lf: Second. INCREASE it for 0.289s improvement per unit
        
        'most_impactful': ('tire_psi_rr', 0.551),  # Top finding
        
        'r_squared': 0.823  # Model explains 82.3% of lap time variance
        # Higher R² = more confident in coefficients
    },
    
    # ===== PRESERVED =====
    'recommendation': None,  # Agent 3 hasn't run yet
    'error': None
}
```

**Analysis Method Decision Tree:**

```
Sample Size Check?
├─ < 10 samples
│  └─ Use: CORRELATION (simpler, more robust)
│
├─ >= 10 samples
│  ├─ Feature Count Check?
│  │  ├─ features > samples/2
│  │  │  └─ Use: CORRELATION (feature-to-sample ratio too high)
│  │  │
│  │  ├─ features <= samples/2
│  │  │  ├─ Variance Check?
│  │  │  │  ├─ variance < 0.15 seconds
│  │  │  │  │  └─ Use: CORRELATION (low variance = weak signal)
│  │  │  │  │
│  │  │  │  └─ variance >= 0.15 seconds
│  │  │  │     └─ Use: REGRESSION (adequate data for modeling)

CORRELATION Method:
├─ For each feature: calculate Pearson correlation with fastest_time
├─ Result range: -1.0 (strong negative) to +1.0 (strong positive)
├─ Interpretation:
│  ├─ -0.8 to -1.0: Increase feature → faster lap
│  ├─ -0.3 to -0.8: Increase feature → moderately faster lap
│  ├─ 0 to 0.3: Weak relationship
│  └─ +0.3 to +1.0: Increase feature → slower lap

REGRESSION Method:
├─ Standardize features (zero mean, unit variance)
├─ Fit linear model: fastest_time = intercept + b1*feature1 + b2*feature2 + ...
├─ Result: standardized regression coefficients
├─ Interpretation:
│  ├─ Coefficient = change in lap time per standard deviation of feature
│  ├─ Negative = increase feature → faster lap
│  ├─ Positive = increase feature → slower lap
│  └─ |Coefficient| = magnitude of impact
```

---

### 4. After Agent 3: Crew Chief (race_engineer.py, lines 350-513)

**Final state from engineer_agent():**

```python
{
    # ===== COMPLETE STATE PRESERVED =====
    'raw_setup_data': DataFrame(...),
    'driver_feedback': {...},
    'driver_diagnosis': {...},
    'data_quality_decision': 'removed_3_outliers',
    'selected_features': [...],
    'analysis_strategy': 'regression',
    'analysis': {...},
    
    # ===== AGENT 3 OUTPUT =====
    'recommendation': """PRIMARY FOCUS: REDUCE tire_psi_rr
        Predicted impact: 0.551s per standardized unit
        Confidence: High (regression coefficient)
        
        Addresses driver complaint: Oversteer (loose rear end)
        
        Rationale: Driver reported loose rear and data analysis confirms 
        tire_psi_rr is the primary factor. Reducing rear tire pressure 
        improves rear grip, reducing oversteer."""
    
    # ===== NO ERRORS =====
    'error': None
}
```

**Agent 3 Decision Flow:**

```
Signal Strength Assessment
├─ |impact| > 0.1?  →  STRONG signal
├─ 0.05 < |impact| <= 0.1?  →  MODERATE signal
└─ |impact| <= 0.05?  →  WEAK signal

Driver vs. Data Conflict Resolution
├─ Does top parameter match driver priority_features?
│  ├─ YES: "driver_validated_by_data" (highest confidence)
│  └─ NO: Check if ANY priority features correlate well
│      ├─ YES: "driver_feedback_prioritized" 
│      │        (driver has physical feel we don't capture)
│      └─ NO: "data_prioritized_no_alternatives"
│
└─ If no driver feedback: "data_only"

Recommendation Formulation
├─ STRONG + driver_validated  →  Single-parameter, high confidence
├─ MODERATE + driver_feedback  →  Single-parameter with caveats
└─ WEAK signal  →  Multi-parameter interaction testing recommended

Generate LLM Explanation (if API available)
├─ Prompt includes: driver complaint, data findings, priority features
├─ Temperature: 0.3 (mostly deterministic, slight variation)
└─ Output: Structured bullets (SITUATION, DECISION, IMPACT, NEXT STEPS)
```

---

## Driver Feedback Data Structure

### Natural Language Input
```
User: "Car feels loose coming off the corners, rear end wants to come around when 
I get on the throttle. Especially bad in turns 1 and 2."
```

### LLM-Interpreted Structure (driver_feedback_interpreter.py)
```python
{
    'complaint': 'loose_exit',  # Options from controlled vocabulary:
                                # - loose_oversteer, tight_understeer
                                # - loose_entry, loose_exit
                                # - tight_entry, tight_exit
                                # - bottoming, chattering
                                # - general
    
    'severity': 'moderate',     # Options: minor, moderate, severe
    
    'phase': 'corner_exit',     # When does it happen?
                                # Options: corner_entry, corner_exit, mid_corner,
                                #         straightaway, all_phases
    
    'diagnosis': 'Insufficient rear grip causing oversteer on throttle application',
    
    'priority_features': [      # These parameters might fix the issue
        'tire_psi_rr',          # Most likely
        'tire_psi_lr',          # Second
        'track_bar_height_left' # Third
    ]
}
```

### Rule-Based Fallback (if LLM unavailable)
```python
# Keywords mapped to complaints:
{
    'loose|oversteer|rear end|spin': 'loose_oversteer',
    'tight|understeer|push|front end': 'tight_understeer',
    'bottom|hitting|harsh': 'bottoming'
}

# Phase detection:
if 'exit' in feedback or 'throttle' in feedback:
    phase = 'corner_exit'
elif 'entry' in feedback or 'turn in' in feedback:
    phase = 'corner_entry'
else:
    phase = 'mid_corner'

# Severity detection:
if any(word in feedback for word in ['really', 'very', 'bad']):
    severity = 'severe'
elif any(word in feedback for word in ['slight', 'little', 'bit']):
    severity = 'minor'
else:
    severity = 'moderate'
```

---

## Analysis Results Structure

### Correlation Analysis Output
```python
analysis = {
    'method': 'correlation',
    
    'all_impacts': {
        'tire_psi_rr': -0.751,   # Strong negative: reduce it
        'tire_psi_lf': 0.623,    # Positive: increase it helps
        'cross_weight': -0.289,  # Moderate negative
        # ... more parameters
    },
    
    'sorted_impacts': [
        ('tire_psi_rr', -0.751),
        ('tire_psi_lf', 0.623),
        ('cross_weight', -0.289),
        # ...
    ],
    
    'most_impactful': ('tire_psi_rr', -0.751)
    
    # Note: No r_squared (not applicable to correlation)
}
```

### Regression Analysis Output
```python
analysis = {
    'method': 'regression',
    
    'all_impacts': {
        'tire_psi_rr': 0.551,      # Standardized coefficient
        'tire_psi_lf': -0.289,
        'cross_weight': -0.195,
        # ...
    },
    
    'sorted_impacts': [
        ('tire_psi_rr', 0.551),
        ('tire_psi_lf', -0.289),
        ('cross_weight', -0.195),
        # ...
    ],
    
    'most_impactful': ('tire_psi_rr', 0.551),
    
    'r_squared': 0.823  # Model explains 82.3% of variance
}
```

**Interpretation:**
- Positive coefficient = increase feature → slower lap (worse)
- Negative coefficient = increase feature → faster lap (better)
- |coefficient| = magnitude of impact
- Standardized = per standard deviation of feature

---

## Output Files Structure

### recommendations.json
```json
{
    "timestamp": "2025-11-07T14:30:45",
    "data_source": "mock_data",
    "recommendation": "PRIMARY FOCUS: REDUCE tire_psi_rr...",
    "analysis": {
        "method": "regression",
        "all_impacts": {
            "tire_psi_rr": 0.551,
            "tire_psi_lf": -0.289
        },
        "most_impactful": ["tire_psi_rr", 0.551]
    },
    "best_time": 15.213,
    "baseline_time": 15.543,
    "improvement": 0.33,
    "num_sessions": 20
}
```

---

## Data Type Mappings

### Setup Parameters
| Parameter | Type | Units | Range | Notes |
|-----------|------|-------|-------|-------|
| tire_psi_* | float | PSI | 20-35 | 4 tires (LF, RF, LR, RR) |
| cross_weight | float | % | 45-55 | Weight distribution front-left to rear-right |
| track_bar_height_left | float | mm | 0-20 | Anti-roll bar height |
| spring_* | int | N/mm | 200-500 | 4 springs (LF, RF, LR, RR) |
| ride_height_* | float | mm | 50-100 | 4 corners |

### Performance Metrics
| Metric | Type | Units | Notes |
|--------|------|-------|-------|
| fastest_time | float | seconds | Best lap in session |
| lap_time | float | seconds | Individual lap duration |
| speed_avg | float | mph | Average speed over lap |
| tire_temp_* | float | F | 4 corners (LF, RF, LR, RR) |
| lat_accel_max | float | G | Peak lateral acceleration |

### Agent Outputs
| Field | Type | Values | Notes |
|-------|------|--------|-------|
| complaint | str | enum | Controlled vocabulary |
| severity | str | {minor, moderate, severe} | Impact assessment |
| phase | str | enum | When issue occurs |
| method | str | {correlation, regression} | Analysis technique |
| signal_strength | str | {STRONG, MODERATE, WEAK} | Confidence level |

