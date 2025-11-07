# State Management & Context Improvements

## Overview

This document describes enhancements to ensure all driver input is incorporated into decision-making and to integrate setup manual knowledge into the AI Race Engineer system.

## Problems Solved

### Problem 1: Context Loss in Driver Feedback
**Issue:** Driver mentions constraints like "RR tire PSI is as low as legally allowed" but agents still recommend reducing it.

**Root Cause:** LLM interpretation converted natural language to structured data, but constraint information was discarded. Agents only saw the diagnosis, not the original message.

**Solution:**
1. Preserve raw driver feedback in state
2. Extract constraints separately
3. Validate recommendations against constraints
4. Find alternatives when constraints violated

### Problem 2: No Setup Manual Knowledge
**Issue:** iRacing provides a car setup manual (PDF) with legal limits, handling guides, and best practices, but it wasn't being utilized.

**Root Cause:** No system to load, parse, and reference external documentation.

**Solution:**
1. PDF knowledge base loader
2. Parameter limit extraction
3. Handling guide integration
4. Context injection into LLM recommendations

---

## Implementation Details

### 1. Constraint Extraction System

**New File:** `constraint_extractor.py`

**Features:**
- **LLM-based extraction** (primary): Uses Claude Haiku to parse constraints from natural language
- **Rule-based extraction** (fallback): Pattern matching for common phrases
- **Three constraint types:**
  - `parameter_limits`: Parameters at min/max (e.g., "as low as allowed")
  - `already_tried`: Parameters already adjusted (e.g., "we already tried increasing...")
  - `cannot_adjust`: Parameters that can't be changed (e.g., "locked in")

**Example:**
```python
from constraint_extractor import extract_constraints

feedback = "RR tire PSI is as low as legally allowed. Already tried springs."
constraints = extract_constraints(feedback)

# Output:
# {
#     'parameter_limits': [
#         {'param': 'tire_psi_rr', 'limit_type': 'at_minimum', 'reason': 'as low as legally allowed'}
#     ],
#     'already_tried': ['spring_rr'],
#     'cannot_adjust': [],
#     'raw_constraints': ["RR tire PSI is as low as legally allowed"]
# }
```

**Pattern Detection:**
- "as low as allowed/possible" → `at_minimum`
- "maxed out" / "as high as possible" → `at_maximum`
- "already tried/tested" → `already_tried`
- "can't adjust/change" → `cannot_adjust`

---

### 2. Enhanced State Management

**Modified:** `race_engineer.py` - RaceEngineerState class

**New State Fields:**
```python
# Driver Context Fields (preserve all driver input)
raw_driver_feedback: Optional[str]      # Original driver message (unprocessed)
driver_constraints: Optional[Dict]      # Extracted limits and constraints
setup_knowledge_base: Optional[Dict]    # Car setup manual context
```

**Benefits:**
- **No information loss**: Original feedback preserved
- **Explicit constraints**: Agents can validate recommendations
- **Knowledge access**: Setup manual available to all agents

**State Flow:**
```
Demo.py:
  ├─ raw_driver_feedback = input()
  ├─ driver_feedback = interpret_with_llm(raw)          # Structured diagnosis
  ├─ driver_constraints = extract_constraints(raw)      # Constraint extraction
  └─ setup_knowledge_base = load_setup_manual()         # Knowledge base

Initial State = {
    raw_driver_feedback,      # "RR tire PSI is as low as allowed"
    driver_feedback,          # {complaint: "loose_exit", ...}
    driver_constraints,       # {parameter_limits: [...], ...}
    setup_knowledge_base,     # {parameter_limits, handling_guides, ...}
    ...
}

Agent 1 → Agent 2 → Agent 3
                     └─ Validates against constraints
                     └─ Consults knowledge base
                     └─ Finds alternatives if needed
```

---

### 3. Agent 3 Constraint Validation

**Modified:** `race_engineer.py` - engineer_agent() function

**New Decision Point: DECISION 0.5 - Validate Against Constraints**

**Logic:**
1. Check if recommended parameter has a constraint
2. Determine if recommendation violates constraint:
   - Recommending REDUCE but param is `at_minimum` → VIOLATION
   - Recommending INCREASE but param is `at_maximum` → VIOLATION
3. If violated:
   - Print warning
   - Find next best parameter that doesn't violate constraints
   - Switch recommendation

**Example Output:**
```
[AGENT 3] Crew Chief: Synthesizing recommendations...
   [TARGET] Top parameter from data: tire_psi_rr
   [STATS] Impact magnitude: 0.551

   [CONSTRAINT] WARNING: tire_psi_rr is at_minimum - as low as legally allowed
   DECISION: Finding alternative parameter...
   ALTERNATIVE: Recommending tire_psi_lr instead (impact: 0.322)
```

**Code Snippet:**
```python
# DECISION 0.5: Validate against driver constraints
driver_constraints = state.get('driver_constraints', {})

if driver_constraints:
    parameter_limits = driver_constraints.get('parameter_limits', [])

    for limit in parameter_limits:
        if limit['param'] == param:
            limit_type = limit['limit_type']
            recommending_increase = (method == "correlation" and impact < 0) or \
                                   (method == "regression" and impact < 0)

            # Check if recommendation violates limit
            if (limit_type == "at_minimum" and not recommending_increase) or \
               (limit_type == "at_maximum" and recommending_increase):
                # Find alternative parameter
                for alt_param, alt_impact in sorted_impacts:
                    if not violates_constraints(alt_param):
                        param = alt_param
                        impact = alt_impact
                        break
```

---

### 4. Knowledge Base System

**New File:** `knowledge_base_loader.py`

**Features:**
- **PDF Loading**: Extracts text from iRacing setup manual PDFs (requires PyPDF2)
- **Default Knowledge**: Built-in NASCAR setup knowledge (fallback)
- **Structured Data:**
  - `parameter_limits`: Legal ranges (e.g., tire_psi_rr: 18-40 psi)
  - `handling_guides`: Recommendations by issue type (loose_exit, tight_understeer, etc.)
  - `parameter_interactions`: Known synergies and conflicts
  - `indexed_content`: Searchable manual sections

**Usage:**
```python
from knowledge_base_loader import load_setup_manual, get_relevant_knowledge, format_knowledge_for_llm

# Load manual (tries PDF, falls back to defaults)
kb = load_setup_manual()  # Or load_setup_manual("path/to/manual.pdf")

# Get relevant sections
relevant = get_relevant_knowledge(
    kb,
    handling_issue='loose_exit',
    parameter='tire_psi_rr'
)

# Format for LLM context
context = format_knowledge_for_llm(relevant)
# Output:
# **Parameter Limits:**
#   - tire_psi_rr: 18-40 psi (typical: 28-34)
#
# **Setup Guidance:**
#   Car is loose (oversteer) on corner exit
#   Recommended changes:
#     1. tire_psi_rr - increase (Primary adjustment for exit oversteer)
#     2. tire_psi_lr - increase (Secondary rear grip adjustment)
#   ⚠️ Rear tire pressure is most effective but has narrow window
```

**Default NASCAR Knowledge Includes:**

| Aspect | Details |
|--------|---------|
| **Parameter Limits** | Legal ranges for 10+ parameters (tire PSI, cross weight, springs, etc.) |
| **Handling Guides** | Recommendations for 4 common issues (loose_exit, loose_entry, tight_understeer, bottoming) |
| **Parameter Interactions** | How parameters affect each other (e.g., tire PSI affects how springs work) |
| **Best Practices** | Testing approach, change sizes, typical ranges |

---

### 5. Demo.py Integration

**Modified:** `demo.py`

**New Steps:**

**Step 1.4 - Load Knowledge Base:**
```python
print("[1.4/5] Loading setup knowledge base...")
from knowledge_base_loader import load_setup_manual

setup_knowledge_base = load_setup_manual()
# Tries ./docs/setup_manual.pdf, falls back to defaults
```

**Step 1.75 - Extract Constraints:**
```python
print("   [AI] Checking for setup constraints...")
from constraint_extractor import extract_constraints

driver_constraints = extract_constraints(raw_driver_feedback)

if driver_constraints.get('parameter_limits'):
    print(f"   [CONSTRAINTS] Found {len(parameter_limits)} constraint(s):")
    for limit in parameter_limits:
        print(f"      • {limit['param']}: {limit['limit_type']}")
```

**State Initialization:**
```python
initial_state = {
    # ... existing fields ...

    # Driver context fields (NEW)
    'raw_driver_feedback': raw_driver_feedback,
    'driver_constraints': driver_constraints,
    'setup_knowledge_base': setup_knowledge_base
}
```

---

## Benefits & Impact

### 1. Complete Context Preservation
- **Before:** "RR tire PSI is as low as allowed" → Lost in interpretation
- **After:** Preserved in `raw_driver_feedback`, extracted to `driver_constraints`

### 2. Intelligent Constraint Handling
- **Before:** Recommends "REDUCE tire_psi_rr" even when driver says it's at minimum
- **After:** Detects violation, finds alternative (tire_psi_lr), explains decision

### 3. Knowledge-Informed Recommendations
- **Before:** Pure data-driven, no awareness of legal limits or best practices
- **After:** Consults setup manual, shows legal ranges, cites handling guides

### 4. Better Decision Transparency
```
[CONSTRAINT] WARNING: tire_psi_rr is at_minimum
DECISION: Finding alternative parameter...
ALTERNATIVE: Recommending tire_psi_lr instead

[KNOWLEDGE BASE] Consulting setup manual...
   tire_psi_lr: 18.0-35.0 psi (typical: 24-28)
```

---

## Usage Guide

### For Users: Adding Your Setup Manual

1. **Place PDF in docs/ directory:**
   ```bash
   mkdir -p docs
   cp ~/Downloads/iracing_setup_guide.pdf docs/setup_manual.pdf
   ```

2. **Install PDF support (optional):**
   ```bash
   pip install PyPDF2
   ```

3. **Run normally:**
   ```bash
   python demo.py "Your feedback with constraints"
   ```

   The system will:
   - Detect and load your PDF
   - Extract parameter limits
   - Use manual content in recommendations

4. **Fallback:** If no PDF or PyPDF2 not installed, uses built-in NASCAR knowledge

### For Users: Providing Constraint Feedback

The system now understands:

✅ **Parameter Limits:**
- "RR tire pressure is as low as legally allowed"
- "Cross weight is maxed out at 56%"
- "Can't go any higher on spring rates"

✅ **Already Tried:**
- "We already tried increasing cross weight"
- "Already tested rear tire pressure changes"

✅ **Cannot Adjust:**
- "Can't adjust the track bar further"
- "Front springs are locked in"

**Example:**
```bash
python demo.py "Car is loose on exit. RR tire PSI is already at 18 psi (minimum). We tried springs last run but it didn't help."
```

Output:
```
[CONSTRAINTS] Found 2 constraint(s):
   • tire_psi_rr: at_minimum
   • Already tried: spring_rr

[AGENT 3] DECISION: Finding alternative...
   ALTERNATIVE: Recommending tire_psi_lr instead
```

---

## Technical Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT                                │
│  "RR tire PSI as low as allowed, still loose on exit"       │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ LLM Interpret │   │ Extract          │   │ Load Knowledge  │
│ Feedback      │   │ Constraints      │   │ Base            │
└──────┬────────┘   └──────┬───────────┘   └────────┬────────┘
       │                   │                         │
       │                   │                         │
       ▼                   ▼                         ▼
   driver_feedback   driver_constraints    setup_knowledge_base
       │                   │                         │
       └───────────────────┴─────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ Initial State  │
                  └────────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     ┌────────┐      ┌────────┐      ┌────────┐
     │Agent 1 │─────▶│Agent 2 │─────▶│Agent 3 │
     │        │      │        │      │        │
     └────────┘      └────────┘      └────┬───┘
                                          │
                                          │ Validate against
                                          │ constraints
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │ If violated: │
                                   │ Find         │
                                   │ alternative  │
                                   └──────┬───────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │ Consult KB   │
                                   │ for limits   │
                                   │ & guidance   │
                                   └──────┬───────┘
                                          │
                                          ▼
                                   Recommendation
```

---

## Files Modified/Created

| File | Type | Changes |
|------|------|---------|
| `constraint_extractor.py` | NEW | LLM + rule-based constraint extraction |
| `knowledge_base_loader.py` | NEW | PDF parser + NASCAR knowledge base |
| `race_engineer.py` | MODIFIED | +3 state fields, constraint validation in Agent 3 |
| `demo.py` | MODIFIED | Load KB, extract constraints, add to state |
| `STATE_MANAGEMENT_IMPROVEMENTS.md` | NEW | This documentation |

---

## Testing & Validation

### Test Case 1: Constraint Detection
**Input:**
```
"Car feels loose. RR tire PSI is as low as legally allowed."
```

**Expected Behavior:**
- ✅ Detects constraint: `tire_psi_rr at_minimum`
- ✅ Agent 3 catches violation
- ✅ Switches to alternative parameter
- ✅ Explains decision

**Actual Result:** ✅ PASS

### Test Case 2: Knowledge Base Integration
**Input:**
```
"Loose on exit"
```

**Expected Behavior:**
- ✅ Loads knowledge base (PDF or defaults)
- ✅ Retrieves handling guide for `loose_exit`
- ✅ Shows parameter limits
- ✅ Includes in LLM context

**Actual Result:** ✅ PASS

### Test Case 3: Multi-Constraint Handling
**Input:**
```
"RR tire at minimum, LR already tested, can't adjust springs"
```

**Expected Behavior:**
- ✅ Extracts 3 constraints
- ✅ Validates all before recommending
- ✅ Finds parameter that satisfies all constraints

**Actual Result:** ✅ PASS (requires API key for full LLM extraction)

---

## Future Enhancements

### 1. Outcome Validation Loop
**Goal:** Learn from recommendation effectiveness

**Approach:**
- After driver applies change and runs laps, collect feedback:
  ```python
  session_mgr.add_outcome_feedback(session_id, {
      'lap_time_improvement': -0.3,  # seconds
      'driver_assessment': 'improved',
      'parameter_changed': 'tire_psi_lr',
      'new_value': 26.5
  })
  ```
- Agent 3 checks outcome history before recommending
- Downgrade confidence if parameter repeatedly fails

### 2. Interaction Analysis
**Goal:** Detect multi-parameter synergies

**Approach:**
- When single-parameter changes weak, test interactions
- Use polynomial regression or decision trees
- Example: "tire_psi_rr + spring_rr work together"

### 3. Advanced Knowledge Base
**Goal:** Deeper PDF parsing and indexing

**Approach:**
- Semantic search over manual sections
- Extract tables and charts
- Link recommendations to specific manual pages

### 4. Constraint Prediction
**Goal:** Proactively warn about limits

**Approach:**
- Track parameter history: [30.5, 30.0, 29.5, 29.0, ...]
- Predict approaching limit: "RR tire approaching minimum"
- Suggest transition strategy before hitting constraint

---

## Summary

This enhancement addresses the critical issue of context loss in driver feedback and adds a comprehensive knowledge base system. The AI Race Engineer now:

1. **Preserves all driver input** - No information lost during interpretation
2. **Extracts and validates constraints** - Won't recommend impossible changes
3. **Consults setup manual** - Legal limits and best practices inform decisions
4. **Finds intelligent alternatives** - When constraints block primary recommendation
5. **Explains decisions transparently** - Shows constraint violations and alternatives

**Impact:** System now respects real-world constraints and makes feasible recommendations backed by setup manual knowledge.
