# AI Race Engineer Codebase Architecture Summary

## 1. SYSTEM OVERVIEW

This is an agentic NASCAR race engineering system that uses LangGraph to orchestrate three specialized AI agents that process lap data and generate setup recommendations. The system integrates real telemetry data (from .ibt and .ldx files) with LLM capabilities for natural language understanding.

**Core Value Proposition:**
- Interprets driver feedback (natural language) → Technical diagnosis
- Analyzes lap/setup data correlations → Quantified impacts
- Generates setup recommendations with decision rationale
- Validates driver intuition with data analysis

---

## 2. DATA STRUCTURES & STATE MANAGEMENT

### Main State Definition (race_engineer.py, lines 18-27)

```python
class RaceEngineerState(TypedDict):
    raw_setup_data: Optional[pd.DataFrame]           # Session-level data
    driver_feedback: Optional[Dict]                   # Natural language feedback
    driver_diagnosis: Optional[Dict]                  # Agent 1's interpretation
    data_quality_decision: Optional[str]              # Outlier/validity decision
    analysis_strategy: Optional[str]                  # Strategy choice (correlation/regression)
    selected_features: Optional[List[str]]            # Features Agent 2 selected
    analysis: Optional[Dict]                          # Results from Agent 2
    recommendation: Optional[str]                     # Final recommendation from Agent 3
    error: Optional[str]                              # Error tracking
```

### Data Row Format (session-level)
Each row represents one test session with:
- **Setup Parameters:** tire_psi_lf, tire_psi_rf, tire_psi_lr, tire_psi_rr, cross_weight, track_bar_height_left, spring_lf, spring_rf, etc.
- **Performance:** fastest_time (seconds)
- **Session Info:** session_id, venue, etc.

### Key State Transition Points
1. **Agent 1 → Agent 2:** Adds `driver_diagnosis`, `data_quality_decision`
2. **Agent 2 → Agent 3:** Adds `analysis`, `selected_features`, `analysis_strategy`
3. **Agent 3 → Output:** Adds `recommendation`

---

## 3. AGENT ARCHITECTURE

### Agent 1: Telemetry Chief (Data Quality + Driver Feedback)
**File:** race_engineer.py, lines 67-185
**Responsibilities:**
- Interprets driver feedback (complaint → technical diagnosis)
- Assesses data quality (outlier detection, sample size validation)
- Makes visibility-first decisions

**Key Decision Points:**
1. What is driver feeling? (uses complaint keywords to map to priority features)
2. Should we remove outliers? (IQR method, < 20% threshold)
3. Is data sufficient? (requires ≥5 sessions)

**Output:**
```python
{
    "driver_diagnosis": {
        'diagnosis': "Oversteer (loose rear end)",
        'technical_cause': "Insufficient rear grip...",
        'priority_features': ['tire_psi_rr', 'tire_psi_lr', 'track_bar_height_left'],
        'complaint_type': 'loose_exit'
    },
    "data_quality_decision": "removed_3_outliers"  # or "no_outliers_found"
}
```

### Agent 2: Data Scientist (Feature Selection + Model Strategy)
**File:** race_engineer.py, lines 188-347
**Responsibilities:**
- Dynamic feature selection (variance-based filtering)
- Strategic analysis method selection
- Correlation or regression analysis execution

**Key Decision Points:**
1. Which features have meaningful variance? (std > 0.01 threshold)
2. Correlation or regression? (based on sample size, feature count, variance)
3. Prioritize driver feedback parameters? (weight them in analysis)

**Analysis Methods:**
- **Correlation:** Simple correlations for small datasets
- **Regression:** Scaled linear regression with StandardScaler

**Output:**
```python
{
    "analysis": {
        'method': 'regression' | 'correlation',
        'all_impacts': {'tire_psi_rr': 0.551, 'cross_weight': -0.289, ...},
        'most_impactful': ('tire_psi_rr', 0.551),
        'r_squared': 0.823  # for regression only
    },
    "selected_features": ['tire_psi_lf', 'tire_psi_rf', 'cross_weight', ...],
    "analysis_strategy": 'regression'
}
```

### Agent 3: Crew Chief (Recommendation Synthesis)
**File:** race_engineer.py, lines 350-513
**Responsibilities:**
- Signal strength assessment (STRONG/MODERATE/WEAK)
- Decision conflict resolution (driver vs. data)
- LLM-powered explanation generation

**Key Decision Points:**
1. Trust driver intuition or data? (checks if top param matches driver priorities)
2. Signal strength? (|impact| > 0.1 = STRONG, > 0.05 = MODERATE)
3. Single-parameter or multi-parameter recommendation?

**Decision Logic:**
- If data-validated driver feedback → High confidence
- If data contradicts driver → Prioritize driver (has physical G-force data)
- If weak signal → Recommend multi-parameter interaction testing

**Output:**
```python
{
    "recommendation": "PRIMARY FOCUS: REDUCE tire_psi_rr\nPredicted impact: 0.551s per standardized unit\nConfidence: High (regression coefficient)\nAddresses driver complaint: Oversteer (loose rear end)"
}
```

---

## 4. LLM INTEGRATION

### Components Using LLM

#### A. Driver Feedback Interpreter (driver_feedback_interpreter.py)
**Purpose:** Convert natural language driver feedback → structured technical diagnosis

**Two Implementations:**
1. **LLM-powered** (Anthropic Claude 3 Haiku)
   - JSON-structured output with complaint type, severity, phase, diagnosis
   - Handles complex/nuanced feedback
   - Falls back to rule-based if API unavailable

2. **Rule-based fallback**
   - Keyword matching for loose/tight/bottoming
   - Maps complaints to priority features
   - Works without API key

**Input:** "Car feels loose off corners but tight on entry"
**Output:**
```json
{
    "complaint": "loose_exit",
    "severity": "moderate",
    "phase": "corner_exit",
    "diagnosis": "Mixed handling characteristics - loose rear on throttle but tight front on entry",
    "priority_features": ["tire_psi_rr", "tire_psi_lr", "track_bar_height_left"]
}
```

#### B. Decision Explanation Generator (llm_explainer.py)
**Purpose:** Generate natural language explanations of Agent 3's decisions

**Features:**
- Single-turn explanation (current decision)
- Multi-turn analysis (learning from session history - not yet implemented)
- Structured bullet format for clarity

**Decision Context Inputs:**
```python
{
    'driver_complaint': 'Oversteer (loose rear end)',
    'data_top_param': 'tire_psi_rr',
    'data_correlation': 0.551,
    'priority_features': ['tire_psi_rr', 'tire_psi_lr'],
    'decision_type': 'driver_validated_by_data',
    'recommended_param': 'tire_psi_rr',
    'recommended_impact': 0.551
}
```

**Output:**
```
• SITUATION: Driver feedback (oversteer) validated by telemetry data
• DECISION: Adjust tire_psi_rr - both driver feel and data point to same root cause
• EXPECTED IMPACT: High confidence change, should improve lap times and driver confidence
• NEXT STEPS: Make adjustment, run 3-5 laps, confirm driver feel matches lap time improvement
```

**Implementation Details:**
- API Key: ANTHROPIC_API_KEY environment variable
- Model: claude-3-haiku-20240307
- Temperature: 0.0-0.3 (deterministic)
- Graceful degradation: Falls back to template-based explanations

---

## 5. DATA PROCESSING PIPELINE

### Entry Points

1. **demo.py** - Interactive entry point
   - Loads/generates data
   - Collects driver feedback (interactive)
   - Runs full workflow
   - Saves results

2. **main.py** - Batch entry point
   - Tests telemetry parsing
   - Generates mock data if needed
   - Runs agents
   - Visualizes results

### Data Loading (csv_data_loader.py)
**Priority:**
1. Try to load .ldx files (MoTeC XML format)
2. Fall back to CSV files
3. Generate mock data if nothing available

**Supported Column Variations:**
- lap_time / LapTime / fastest_time
- session_id / SessionID
- lap_number / LapNumber

**Data Aggregation:**
- If lap-level: Groups by session, takes best lap per session
- If session-level: Uses as-is
- Adds derived metrics: tire_stagger_front, tire_stagger_rear, spring_ratio_front, etc.

### Telemetry Parsing

#### .ldx Files (MoTeC XML Format) - telemetry_parser.py
- Extracts setup parameters from XML
- Converts units (kPa → PSI for tire pressures)
- Calculates derived features (tire stagger, spring ratios, rake)

#### .ibt Files (iRacing Native) - ibt_parser.py
- Reads raw telemetry samples via pyirsdk library
- Extracts channels: Speed, Throttle, Brake, temperatures, accelerations
- Aggregates to lap statistics
- Falls back to mock data if pyirsdk unavailable

---

## 6. WORKFLOW ORCHESTRATION

### LangGraph Structure (race_engineer.py, lines 523-565)

```
START
  ↓
[Telemetry Chief]
  ↓ (check for errors)
[Data Scientist]  or  [Error Handler] → END
  ↓ (check for errors)
[Crew Chief]  or  [Error Handler] → END
  ↓
END (with recommendation)
```

**Conditional Routing:**
- After each agent, checks `state['error']`
- If error exists → Routes to error_handler → END
- Otherwise → Routes to next agent

**State Immutability:**
- Agents don't modify input state, only return new keys
- Merged into existing state: `merged_state = {**state, **updated_state}`
- Previous state preserved for debugging

---

## 7. CURRENT STATE/MEMORY TRACKING

### What's Currently Tracked
✓ Per-session setup parameters and lap times
✓ Analysis results in state dictionary
✓ Decision rationale printed to console

### What's NOT Currently Tracked
✗ Historical analysis across multiple invocations
✗ Previously tested parameter combinations
✗ How effective past recommendations were
✗ Session-to-session learning or pattern detection
✗ Multi-turn conversation context
✗ Recommendation outcomes/validation

### Files That Would Need Modification

1. **race_engineer.py** - RaceEngineerState TypedDict
   - Add session history fields
   - Add learning/memory fields
   - Add validation tracking fields

2. **demo.py** - Main execution flow
   - Add session persistence logic
   - Load previous session data
   - Save current session for future reference

3. **llm_explainer.py** - Already has multi-turn function signature
   - `generate_llm_multi_turn_analysis()` exists but isn't called
   - Would need to pass session history

4. **New File Needed:** session_manager.py or memory_store.py
   - Persist sessions to disk/database
   - Load historical sessions
   - Calculate learning metrics

---

## 8. KEY FILES SUMMARY

| File | Lines | Purpose | Key Classes/Functions |
|------|-------|---------|----------------------|
| race_engineer.py | 626 | Main agent orchestration | RaceEngineerState, telemetry_agent, analysis_agent, engineer_agent, create_race_engineer_workflow |
| demo.py | 226 | Interactive entry point | generate_mock_data, interpret_driver_feedback_with_llm, app.invoke |
| main.py | 301 | Batch entry point | generate_mock_training_data, run_ai_agents, create_visualizations |
| driver_feedback_interpreter.py | 237 | NLP feedback processing | interpret_driver_feedback_with_llm, _call_anthropic, _mock_interpretation |
| llm_explainer.py | 211 | Decision explanations | generate_llm_explanation, generate_llm_multi_turn_analysis |
| csv_data_loader.py | 339 | Data loading | CSVDataLoader, prepare_for_ai_analysis |
| telemetry_parser.py | 233 | MoTeC .ldx parsing | TelemetryParser, parse_ldx_file |
| ibt_parser.py | 351 | iRacing .ibt parsing | IBTParser, TelemetryAggregator |
| race_engineer_enhanced.py | 420 | Alternative agent architecture | data_quality_agent, feature_selection_agent, model_selection_agent, recommendation_agent |

---

## 9. TECHNOLOGY STACK

**Core:**
- Python 3.8+
- pandas - Data manipulation
- LangGraph - Agent orchestration
- scikit-learn - Machine learning (correlation, regression, StandardScaler)
- numpy - Numerical computing

**Optional/Specialized:**
- anthropic - Claude API for NLU and explanations
- pyirsdk - iRacing telemetry reading
- PIL/matplotlib - Visualizations

**Data Formats:**
- CSV - Session-level data
- .ldx (XML) - MoTeC telemetry format
- .ibt (binary) - iRacing native format
- JSON - Configuration and results

---

## 10. EXECUTION FLOW EXAMPLE

```
User runs: python demo.py "Car feels loose off corners"

1. LOAD DATA
   ├─ Try .ldx files
   ├─ Try CSV files
   └─ Generate mock data

2. INTERPRET DRIVER FEEDBACK
   ├─ Call LLM with driver complaint
   └─ Parse response to structured dict

3. RUN AGENT WORKFLOW
   ├─ Agent 1: Data validation + driver diagnosis
   │   ├─ Detect outliers
   │   ├─ Check sample size
   │   └─ Map complaint to priority parameters
   │
   ├─ Agent 2: Feature selection + model strategy
   │   ├─ Filter by variance
   │   ├─ Choose analysis method
   │   └─ Run correlation/regression
   │
   └─ Agent 3: Generate recommendation
       ├─ Assess signal strength
       ├─ Resolve driver vs. data conflicts
       ├─ Call LLM for explanation
       └─ Return structured recommendation

4. SAVE RESULTS
   └─ JSON file with recommendation, analysis, metrics

5. DISPLAY SUMMARY
   └─ Print crew chief perspective to console
```

---

## 11. ARCHITECTURE HIGHLIGHTS

### Separation of Concerns
- **Agent 1:** Domain-agnostic data validation + feedback interpretation
- **Agent 2:** Statistical analysis method selection
- **Agent 3:** Domain-specific race engineering knowledge
- **LLM Layer:** Natural language understanding (Agent 1) and explanation (Agent 3)

### Decision Transparency
- Each agent prints its decisions to console
- State transitions visible via `_print_state_transition()`
- Decision rationale is explicit, not buried in code

### Modular Strategy Pattern
- Analysis strategies are swappable (CorrelationStrategy, RegressionStrategy, DecisionTreeStrategy in race_engineer_enhanced.py)
- Easy to add new analysis methods without changing agent logic

### Graceful Degradation
- LLM calls have fallbacks (template-based explanations)
- API key optional - works without it
- Data can come from .ibt, .ldx, or CSV with automatic format conversion

---

## 12. RECOMMENDATIONS FOR SESSION MEMORY IMPLEMENTATION

### Quick Wins
1. **Persistent Session History** (low effort)
   - Save each run to timestamped JSON files in `sessions/` directory
   - Load previous sessions to show trending improvements

2. **Simple Memory Store** (medium effort)
   - Create SessionStore class to save/load session data
   - Add to demo.py to persist decisions across runs
   - Use SQLite for basic storage

3. **Learning From History** (medium effort)
   - Call `generate_llm_multi_turn_analysis()` (already exists!)
   - Pass last 3 sessions to generate insights
   - Show pattern detection to user

### Architectural Changes Needed
1. Add to RaceEngineerState:
   ```python
   session_history: Optional[List[Dict]]  # Previous sessions
   learning_metrics: Optional[Dict]        # Aggregated patterns
   ```

2. Create session_manager.py module
   ```python
   class SessionManager:
       def save_session(self, session_data, timestamp)
       def load_session_history(self, limit=None)
       def get_learning_metrics(self)
   ```

3. Modify demo.py/main.py to:
   - Load history at startup
   - Add history to initial state
   - Save current session after completion

