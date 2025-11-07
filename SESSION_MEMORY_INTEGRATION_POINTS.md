# Session Memory Integration Points

## Current Architecture (State Flow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CURRENT STATELESS FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

USER INPUT
    │
    ▼
┌─────────────────────────────────────────┐
│ demo.py / main.py                       │
│ • Loads data (CSV, .ldx, mock)          │
│ • Collects driver feedback (optional)   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ RaceEngineerState (in-memory dict)      │
│ • raw_setup_data: DataFrame             │
│ • driver_feedback: Dict                 │
│ • [intermediate states from agents]     │
│ • recommendation: str                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ LangGraph Workflow                      │
│ Agent 1 → Agent 2 → Agent 3             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Output Files                            │
│ • JSON: recommendation + analysis       │
│ • PNG: visualizations                   │
│ • CSV: telemetry processed              │
└─────────────────────────────────────────┘

⚠️  LOST ON PROGRAM EXIT:
    ✗ How effective was the recommendation?
    ✗ What did we test in previous sessions?
    ✗ Are we converging on optimal setup?
    ✗ What patterns emerge across sessions?
```

---

## Proposed Architecture (With Session Memory)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED: STATEFUL WITH MEMORY                           │
└─────────────────────────────────────────────────────────────────────────────┘

USER INPUT
    │
    ▼
┌─────────────────────────────────────────┐
│ demo.py / main.py (MODIFIED)            │
│ • Load SESSION HISTORY (new)            │
│ • Loads current data                    │
│ • Collects driver feedback              │
└──────────────┬──────────────────────────┘
               │
               ├─────────────────────────────────┐
               │                                 │
               ▼                                 ▼
    ┌──────────────────────┐    ┌───────────────────────┐
    │ Session History      │    │ Learning Metrics      │
    │ (NEW PERSISTENT)     │    │ (NEW PERSISTENT)      │
    │ • Past 10 sessions   │    │ • Parameter trends    │
    │ • Recommendations    │    │ • Convergence rate    │
    │ • Outcomes/feedback  │    │ • Most effective      │
    └──────────────┬───────┘    └──────────┬────────────┘
                   │                       │
                   └───────────┬───────────┘
                               │
                               ▼
               ┌─────────────────────────────────┐
               │ SESSION_MANAGER                 │
               │ (NEW MODULE)                    │
               │                                 │
               │ Methods:                        │
               │ • load_session_history()        │
               │ • get_learning_metrics()        │
               │ • add_previous_context()        │
               └────────────────┬────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────┐
│ RaceEngineerState (ENHANCED)            │
│ • raw_setup_data: DataFrame             │
│ • driver_feedback: Dict                 │
│ • session_history: List[Dict] (NEW)     │
│ • learning_metrics: Dict (NEW)          │
│ • [intermediate states from agents]     │
│ • recommendation: str                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ LangGraph Workflow (ENHANCED)           │
│ Agent 1 → Agent 2 → Agent 3             │
│                         │               │
│                         ├─→ Consider    │
│                         │   prior       │
│                         │   sessions    │
└──────────────┬──────────────────────────┘
               │
               ├──→ LLM Multi-Turn Analysis (NEW)
               │    • generate_llm_multi_turn_analysis()
               │    • Pattern detection
               │    • Session-to-session learning
               │
               ▼
┌─────────────────────────────────────────┐
│ Output Files (EXPANDED)                 │
│ • JSON: recommendation + analysis       │
│ • JSON: session outcome/feedback (NEW)  │
│ • PNG: visualizations                   │
│ • CSV: session history (NEW)            │
│ • PNG: convergence trends (NEW)         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Persistent Storage (NEW)                │
│ • sessions/ directory                   │
│ • session_{timestamp}.json              │
│ • learning_metrics.json                 │
└─────────────────────────────────────────┘

✓ Retained across sessions:
  ✓ Setup parameter evolution
  ✓ Recommendation effectiveness
  ✓ Convergence to optimal setup
  ✓ Pattern recognition across test days
  ✓ Driver feedback consistency checks
```

---

## Specific Integration Points

### 1. RaceEngineerState Enhancement

**File:** `race_engineer.py` (lines 18-27)

**Current:**
```python
class RaceEngineerState(TypedDict):
    raw_setup_data: Optional[pd.DataFrame]
    driver_feedback: Optional[Dict]
    driver_diagnosis: Optional[Dict]
    data_quality_decision: Optional[str]
    analysis_strategy: Optional[str]
    selected_features: Optional[List[str]]
    analysis: Optional[Dict]
    recommendation: Optional[str]
    error: Optional[str]
```

**Proposed Addition:**
```python
class RaceEngineerState(TypedDict):
    # ... existing fields ...
    
    # NEW: Session Memory Fields
    session_history: Optional[List[Dict]]      # Previous sessions data
    session_timestamp: Optional[str]            # ISO timestamp for this session
    learning_metrics: Optional[Dict]            # Aggregated patterns across sessions
    previous_recommendations: Optional[List]    # Last 3-5 recommendations
    outcome_feedback: Optional[Dict]            # User validation of recommendation
    convergence_progress: Optional[float]       # % improvement trend
```

---

### 2. Session Manager Module

**File:** NEW `session_manager.py`

```python
class SessionManager:
    """
    Manages persistent session storage and retrieval
    Integrates with race_engineer workflow
    """
    
    def __init__(self, storage_dir: Path = Path("sessions")):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(exist_ok=True)
    
    def save_session(self, state: RaceEngineerState, session_id: str):
        """
        Save completed session to persistent storage
        Called after engineer_agent completes
        
        Saves:
        - Input data summary
        - Driver feedback
        - Analysis results
        - Recommendation
        - Timestamp
        """
    
    def load_session_history(self, limit: int = 5) -> List[Dict]:
        """
        Load previous sessions for context
        Called at demo/main startup
        
        Returns:
        - Last N sessions (default 5)
        - Summary format (not full DataFrames)
        """
    
    def get_learning_metrics(self) -> Dict:
        """
        Calculate aggregated patterns across sessions
        Called after loading history
        
        Returns:
        - Parameter convergence trends
        - Recommendation effectiveness rates
        - Most impactful parameters historically
        - Confidence scores from previous tests
        """
    
    def add_outcome_feedback(self, session_id: str, feedback: Dict):
        """
        Record how effective a recommendation was
        Called in extended demo with validation
        
        Stores:
        - Lap time improvement
        - Driver's assessment
        - Parameter validation results
        """
```

---

### 3. demo.py Integration Points

**File:** `demo.py` (currently 226 lines)

**Location 1: After data loading (line ~75)**
```python
# Load data
df = loader.load_data()

# NEW: Load session history
session_mgr = SessionManager()
session_history = session_mgr.load_session_history(limit=5)
learning_metrics = session_mgr.get_learning_metrics()

print(f"\n[MEMORY] Previous sessions: {len(session_history)}")
if learning_metrics:
    print(f"[LEARNING] Most tested parameter: {learning_metrics['most_tested_param']}")
```

**Location 2: Before agents run (line ~152)**
```python
initial_state = {
    'raw_setup_data': df,
    'driver_feedback': driver_feedback,
    'session_history': session_history,        # NEW
    'learning_metrics': learning_metrics,      # NEW
    'session_timestamp': pd.Timestamp.now().isoformat(),  # NEW
    # ... other fields ...
}
```

**Location 3: After agents complete (line ~186)**
```python
# NEW: Save session for future reference
session_mgr.save_session(state, session_id=f"session_{timestamp}")

# NEW: Call multi-turn LLM analysis if we have history
if session_history:
    insights = generate_llm_multi_turn_analysis(
        session_history,
        state.get('recommendation', {})
    )
    print(f"\n[LEARNING] Cross-session insights:")
    print(insights)
```

---

### 4. LLM Explainer Enhancement

**File:** `llm_explainer.py` (has structure, needs calling)

**Current State (line 111-172):**
```python
def generate_llm_multi_turn_analysis(
    session_history: list,
    current_recommendation: Dict,
    model: str = "claude-3-haiku-20240307"
) -> str:
    """
    This function EXISTS but is NEVER CALLED
    
    Analyzes patterns across sessions:
    - Consistent issues reappearing?
    - Are our changes working?
    - Recommendations for next test?
    """
```

**Integration:** Call this after Agent 3 completes if we have session history

```python
# In engineer_agent (race_engineer.py, after line 504)
try:
    from llm_explainer import generate_llm_multi_turn_analysis
    
    session_history = state.get('session_history', [])
    if session_history and len(session_history) >= 2:
        current_context = {
            'driver_complaint': driver_diagnosis.get('diagnosis', 'None'),
            'recommended_param': param,
            'data_correlation': impact
        }
        
        insights = generate_llm_multi_turn_analysis(
            session_history,
            current_context
        )
        if insights:
            print(f"\n   [LEARNING] Multi-session Analysis:")
            print(f"   {insights}")
except ImportError:
    pass
```

---

### 5. Data Storage Structure

**Directory Layout (NEW):**
```
AI-Race-Engineer/
├── sessions/                          # NEW
│   ├── session_2025-11-07_14-30-45.json
│   ├── session_2025-11-07_15-20-15.json
│   ├── session_2025-11-08_09-15-30.json
│   └── learning_metrics.json          # Aggregated patterns
│
├── race_engineer.py
├── demo.py
├── session_manager.py                 # NEW MODULE
├── CODEBASE_ARCHITECTURE.md
└── ...
```

**Session File Format (JSON):**
```json
{
    "timestamp": "2025-11-07T14:30:45",
    "session_id": "session_2025-11-07_14-30-45",
    "driver_feedback": {
        "complaint": "loose_exit",
        "severity": "moderate",
        "diagnosis": "Oversteer (loose rear end)"
    },
    "analysis_results": {
        "method": "regression",
        "most_impactful": ["tire_psi_rr", 0.551],
        "r_squared": 0.823
    },
    "recommendation": {
        "parameter": "tire_psi_rr",
        "direction": "REDUCE",
        "impact": 0.551,
        "confidence": "HIGH"
    },
    "data_summary": {
        "num_sessions_analyzed": 20,
        "best_lap_time": 15.213,
        "improvement_range": 0.33
    }
}
```

**Learning Metrics File (JSON):**
```json
{
    "total_sessions": 15,
    "date_range": "2025-10-20 to 2025-11-07",
    "most_tested_parameters": {
        "tire_psi_rr": 8,
        "cross_weight": 6,
        "tire_psi_lf": 5
    },
    "most_effective_parameters": {
        "tire_psi_rr": 0.551,
        "cross_weight": -0.289,
        "spring_rf": -0.195
    },
    "recommendation_effectiveness": {
        "implemented_count": 8,
        "positive_outcomes": 7,
        "success_rate": 0.875
    },
    "convergence_metric": 0.78,
    "notes": "Strong convergence toward lower rear tire pressure"
}
```

---

## Implementation Phases

### Phase 1: Minimal (1-2 hours)
1. Create `SessionManager` class with JSON file storage
2. Add save_session() method (simple JSON dump)
3. Add load_session_history() method (read JSON files)
4. Modify demo.py to call save_session() after completion

**Result:** Sessions persist across runs, basic replay capability

### Phase 2: Learning Metrics (2-3 hours)
1. Implement get_learning_metrics() 
2. Add learning_metrics to RaceEngineerState
3. Display metrics in demo output
4. Show "most tested" and "most effective" parameters

**Result:** Users see convergence patterns and optimization progress

### Phase 3: Multi-Turn Analysis (1-2 hours)
1. Call existing generate_llm_multi_turn_analysis()
2. Pass session_history to engineer_agent
3. Display cross-session insights
4. Show pattern detection

**Result:** LLM identifies multi-session patterns (e.g., "rear tire pressure consistently helps")

### Phase 4: Outcome Tracking (2-3 hours)
1. Add interactive outcome feedback collection
2. Record lap time improvements
3. Calculate recommendation effectiveness rates
4. Show confidence scores

**Result:** System learns which recommendations work, adjusts confidence

---

## Code Changes Required (Summary)

| File | Current Lines | Changes | Effort |
|------|---------------|---------|--------|
| race_engineer.py | 626 | Add state fields (10 lines), call multi-turn LLM (10 lines) | 30 min |
| demo.py | 226 | Load/save sessions (30 lines) | 30 min |
| session_manager.py | NEW | New module (150-200 lines) | 1-2 hours |
| csv_data_loader.py | 339 | No changes | 0 |
| llm_explainer.py | 211 | No changes (function exists) | 0 |

**Total Effort:** 2-3 hours for working prototype

---

## Benefits to Demonstrate

1. **Continuous Learning:** System improves recommendations over time
2. **Pattern Recognition:** Identifies which setups work consistently
3. **User Context:** Shows "you tested this before" for parameters
4. **Convergence Tracking:** Visualizes progress toward optimal setup
5. **Recommendation Validation:** Tracks which suggestions actually work
6. **Domain Expertise Building:** LLM learns from session history

---

## Testing Checklist

- [ ] SessionManager saves/loads sessions correctly
- [ ] Learning metrics calculated accurately
- [ ] Multi-turn LLM analysis called when history available
- [ ] UI displays history context clearly
- [ ] No breaking changes to existing workflow
- [ ] Sessions directory created on first run
- [ ] Backward compatible (works with no history)

