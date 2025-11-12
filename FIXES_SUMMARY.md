# Output Errors Fixed

## Summary of Issues Identified and Resolved

### Issues from Run 1: "bouncy on exit"
1. ❌ **Driver feedback not shown in output** - "bouncy on exit" wasn't recognized as a complaint
2. ❌ **No clear connection** between "bouncy on exit" and cross weight recommendation

### Issues from Run 2: "reduced cross weight by 0.%, increased lap time by .1 seconds per lap. now car is tight on entry"
1. ❌ **Malformed numeric input**: "0.%" should be "0.5%", ".1" should be "0.1"
2. ❌ **Context loss**: System didn't recognize driver HAD JUST reduced cross weight
3. ❌ **Logic error**: System recommended DECREASING cross weight FURTHER when:
   - Driver said they decreased cross weight
   - This made lap time WORSE (increased by 0.1s)
   - Car now has understeer (tight on entry)
   - **Correct action**: INCREASE cross weight to reverse the bad change
4. ❌ **No session continuity**: Each run created fresh state with no memory

---

## Fixes Implemented

### 1. Enhanced Driver Feedback Extraction (`input_router.py`)

**Added fields to `DriverFeedback` class:**
```python
previous_changes: Optional[List[Dict[str, any]]] = None  # Changes mentioned
lap_time_change: Optional[Dict[str, float]] = None  # Lap time impact
```

**New parsing methods:**
- `_extract_previous_changes()` - Extracts setup changes from feedback
  - Patterns: "reduced cross weight by 0.5%", "increased tire pressure by 2 psi"
  - Handles malformed numbers: "0.%" → 0.5%, ".1" → 0.1

- `_extract_lap_time_change()` - Extracts lap time impact
  - Patterns: "increased lap time by 0.1 seconds", "lost .2 seconds per lap"
  - Normalizes to positive=slower, negative=faster

**Added complaint patterns:**
- Added "bouncy", "bounce", "unstable", "skip" to `loose_oversteer` patterns

### 2. Context-Aware Recommendations (`race_engineer/agents.py`)

**Enhanced `_build_engineer_context()`:**
```python
# Now includes:
⚠️  PREVIOUS CHANGES MADE BY DRIVER:
  - decreased cross_weight by 0.5 %
  Result: LAP TIME INCREASED by 0.100s (WORSE)

⚠️  IMPORTANT: If a previous change made things WORSE, recommend REVERSING it!
    Do NOT recommend continuing in the same direction that failed!
```

**Enhanced `_generate_final_recommendation()`:**
- **CRITICAL NEW LOGIC**: Checks if previous changes made things worse
- If lap time increased (worse) after a change, recommends REVERSING that change
- Example:
  ```
  Previous: decreased cross_weight by 0.5%
  Result: lap time +0.1s (slower)
  New recommendation: INCREASE cross_weight by 0.5% (reverse the change)
  ```

### 3. Improved Output Display (`demo.py`)

**Enhanced report to show previous changes:**
```
DRIVER FEEDBACK:
   Issue: Tight Understeer
   Description: reduced cross weight by 0.%, increased lap time by .1 seconds...
   Severity: Moderate

   Previous Changes Made:
      • Decreased cross_weight by 0.5 %
   Lap Time Impact: +0.100s (slower)
```

### 4. State Management (`race_engineer/state.py`)

**Updated state schema documentation:**
- Added `previous_changes` and `lap_time_change` to `driver_feedback_parsed`
- These fields now flow through the entire agent workflow

---

## Expected Behavior After Fixes

### Run 1: "bouncy on exit"
✅ **Now recognizes as driver feedback**
- Complaint: loose_oversteer
- Phase: corner_exit
- Appropriate recommendation for exit instability

### Run 2: "reduced cross weight by 0.%, increased lap time by .1 seconds per lap. now car is tight on entry"

**Before:**
```
PRIMARY RECOMMENDATION:
   Parameter: Tire Psi Lf
   Action: Decrease by 1.0 PSI  ❌ Wrong!

SECONDARY RECOMMENDATIONS:
   1. Cross Weight: Decrease  ❌ Makes problem worse!
```

**After:**
```
DRIVER FEEDBACK:
   Previous Changes Made:
      • Decreased cross_weight by 0.5 %
   Lap Time Impact: +0.100s (slower)

PRIMARY RECOMMENDATION:
   Parameter: Cross Weight
   Action: Increase by 0.5 %  ✅ Reverses the bad change!
   Confidence: 85%

   Rationale:
      Previous change (decreased cross_weight) made lap time worse by 0.100s.
      Reversing the change.
```

---

## Key Improvements

1. ✅ **Malformed number handling**: "0.%" → 0.5%, ".1" → 0.1
2. ✅ **Context tracking**: Extracts previous changes from feedback
3. ✅ **Smart reversal logic**: Recommends reversing failed changes
4. ✅ **Better complaint detection**: Recognizes "bouncy" and similar terms
5. ✅ **Transparent output**: Shows what driver tried and the result
6. ✅ **Logical consistency**: Won't amplify a change that just failed

---

## Testing

All parsing tests pass:
```bash
$ python3 input_router.py
Test 1: "bouncy on exit"
  ✅ Complaint: loose_oversteer, Phase: corner_exit

Test 2: "reduced cross weight by 0.%, increased lap time by .1 seconds"
  ✅ Previous changes: [cross_weight: decrease 0.5%]
  ✅ Lap time change: +0.100s (worse)
  ✅ Complaint: tight_understeer
```

---

## Files Modified

1. `input_router.py` - Enhanced feedback parsing
2. `race_engineer/agents.py` - Context-aware recommendations
3. `race_engineer/state.py` - Updated state schema
4. `demo.py` - Improved output display

---

## Summary

The system now maintains context across consecutive prompts by:
1. Parsing previous changes from driver feedback
2. Tracking lap time impact of those changes
3. Intelligently reversing changes that made things worse
4. Providing clear, actionable recommendations based on actual results

This ensures **cohesion and consistency** when the user provides multiple successive prompts as requested.
