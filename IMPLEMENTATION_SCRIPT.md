# Complete Implementation Script

## Files Already Created ✅
1. `/race_engineer/nascar_manual_parser.py` - Parses NASCAR manual
2. `/data/knowledge/nascar_manual_knowledge.json` - Cached knowledge base
3. `/race_engineer/recommendation_deduplicator.py` - Deduplication system
4. `/POLISH_IMPROVEMENTS.md` - Complete analysis document

## Remaining Critical Changes

### 1. Update `race_engineer/state.py` - Add Tracking Fields

**Add to RaceEngineerState class (around line 99)**:

```python
previous_recommendations: List[Dict[str, Any]]
"""History of all recommendations made this session:
   - parameter: str
   - direction: str (increase/decrease)
   - magnitude: float
   - magnitude_unit: str
   - iteration_made: int
   - agent_source: str
   - rationale: str
   - was_included_in_final: bool
"""

parameter_adjustment_history: Dict[str, List[Dict[str, Any]]]
"""Track all adjustments by parameter:
   {
       'tire_psi_rr': [
           {'iteration': 1, 'change': -1.5, 'result': 'accepted'},
           {'iteration': 3, 'change': -1.0, 'result': 'rejected_duplicate'}
       ]
   }
"""

recommendation_stats: Dict[str, Any]
"""Statistics about recommendations:
   - total_proposed: int
   - unique_accepted: int
   - duplicates_filtered: int
   - constraint_violations_caught: int
   - parameters_touched: List[str]
"""
```

**Update `create_initial_state()` function (around line 225)**:

```python
# Add after line 246 "candidate_recommendations": [],
"previous_recommendations": [],
"parameter_adjustment_history": {},
"recommendation_stats": {
    "total_proposed": 0,
    "unique_accepted": 0,
    "duplicates_filtered": 0,
    "constraint_violations_caught": 0,
    "parameters_touched": []
},
```

---

### 2. Update `race_engineer/tools.py` - Integrate NASCAR Manual

**Replace `query_setup_manual` function (around line 508)**:

```python
@tool
def query_setup_manual(issue_type: str, parameter: Optional[str] = None) -> Dict[str, Any]:
    """
    Query NASCAR truck setup knowledge base from parsed manual.

    Now uses actual NASCAR Trucks Manual V6 content!

    Args:
        issue_type: Type of handling issue (oversteer, understeer, etc)
        parameter: Optional specific parameter to get info about

    Returns:
        Dictionary with detailed setup guidance from NASCAR manual
    """
    # Load parsed NASCAR manual knowledge
    knowledge_file = Path(__file__).parent.parent / "data" / "knowledge" / "nascar_manual_knowledge.json"

    if not knowledge_file.exists():
        # Parse manual if not cached
        from race_engineer.nascar_manual_parser import parse_and_cache_manual
        pdf_path = Path(__file__).parent.parent / "NASCAR-Trucks-Manual-V6.pdf"
        knowledge = parse_and_cache_manual(str(pdf_path))
    else:
        with open(knowledge_file, 'r') as f:
            knowledge = json.load(f)

    relevant_sections = []
    principles = []
    parameter_guidance = {}
    fixes = {}

    # Extract relevant handling issue information
    issue_key = issue_type.lower().replace(' ', '_')
    if issue_key in knowledge.get('handling_issues', {}):
        issue_info = knowledge['handling_issues'][issue_key]

        relevant_sections.append(issue_info.get('description', ''))
        principles.extend(issue_info.get('symptoms', []))

        # Get specific fixes from manual
        fixes = issue_info.get('fixes', {})

        # Build parameter guidance from fixes
        for param, fix_info in fixes.items():
            parameter_guidance[param] = {
                'action': fix_info.get('action'),
                'magnitude': fix_info.get('magnitude'),
                'rationale': fix_info.get('rationale'),
                'from_nascar_manual': True
            }

    # Get specific parameter info if requested
    if parameter and parameter in knowledge.get('parameters', {}):
        param_info = knowledge['parameters'][parameter]
        parameter_guidance[parameter] = param_info

    return {
        "relevant_sections": relevant_sections,
        "principles": principles,
        "parameter_guidance": parameter_guidance,
        "fixes": fixes,
        "manual_version": knowledge.get('manual_version', 'V6'),
        "source": "NASCAR Trucks Manual V6"
    }
```

**Replace `check_constraints` function (around line 580)**:

```python
@tool
def check_constraints(
    parameter: str,
    direction: str,
    magnitude: float,
    current_value: Optional[float] = None,
    constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Comprehensive constraint checking against NASCAR manual limits.

    NOW VALIDATES AGAINST ACTUAL NASCAR TRUCKS MANUAL SPECS!

    Checks:
    1. NASCAR rule limits (from manual)
    2. Physical limits
    3. Driver-specified constraints
    4. Safety margins

    Args:
        parameter: Parameter name
        direction: "increase" or "decrease"
        magnitude: Amount of change
        current_value: Optional current value
        constraints: Optional driver constraints

    Returns:
        Dictionary with validation results and limit information
    """
    # Load NASCAR manual constraints
    knowledge_file = Path(__file__).parent.parent / "data" / "knowledge" / "nascar_manual_knowledge.json"

    manual_limits = {}
    if knowledge_file.exists():
        with open(knowledge_file, 'r') as f:
            knowledge = json.load(f)

        # Get parameter limits from manual
        if parameter in knowledge.get('parameters', {}):
            param_info = knowledge['parameters'][parameter]
            if 'range' in param_info:
                manual_limits = {
                    'min': param_info['range']['min'],
                    'max': param_info['range']['max'],
                    'typical_min': param_info.get('typical', {}).get('min'),
                    'typical_max': param_info.get('typical', {}).get('max'),
                    'unit': param_info.get('unit', 'units'),
                    'source': 'NASCAR Trucks Manual V6'
                }

    violations = []
    warnings = []
    proposed_value = None

    # Calculate proposed value if current is provided
    if current_value is not None and manual_limits:
        if direction.lower() == "increase":
            proposed_value = current_value + magnitude
        else:
            proposed_value = current_value - magnitude

        # Check NASCAR manual limits
        if proposed_value < manual_limits['min']:
            violations.append(
                f"{parameter} would be {proposed_value:.2f} {manual_limits['unit']}, "
                f"below NASCAR manual minimum of {manual_limits['min']} {manual_limits['unit']}"
            )
        elif proposed_value > manual_limits['max']:
            violations.append(
                f"{parameter} would be {proposed_value:.2f} {manual_limits['unit']}, "
                f"above NASCAR manual maximum of {manual_limits['max']} {manual_limits['unit']}"
            )

        # Check if approaching limits (within 10%)
        limit_range = manual_limits['max'] - manual_limits['min']
        margin_low = manual_limits['min'] + 0.1 * limit_range
        margin_high = manual_limits['max'] - 0.1 * limit_range

        if proposed_value < margin_low and proposed_value >= manual_limits['min']:
            margin = proposed_value - manual_limits['min']
            warnings.append(
                f"{parameter} approaching minimum limit (margin: {margin:.2f} {manual_limits['unit']})"
            )
        elif proposed_value > margin_high and proposed_value <= manual_limits['max']:
            margin = manual_limits['max'] - proposed_value
            warnings.append(
                f"{parameter} approaching maximum limit (margin: {margin:.2f} {manual_limits['unit']})"
            )

        # Check if outside typical range
        if manual_limits.get('typical_min') and manual_limits.get('typical_max'):
            if proposed_value < manual_limits['typical_min']:
                warnings.append(
                    f"{parameter} below typical range ({manual_limits['typical_min']}-{manual_limits['typical_max']} {manual_limits['unit']})"
                )
            elif proposed_value > manual_limits['typical_max']:
                warnings.append(
                    f"{parameter} above typical range ({manual_limits['typical_min']}-{manual_limits['typical_max']} {manual_limits['unit']})"
                )

    # Check driver constraints if provided
    if constraints:
        params_at_limit = constraints.get('parameters_at_limit', {})
        if parameter in params_at_limit:
            limit_type = params_at_limit[parameter]
            if (limit_type == 'min' and direction.lower() == 'decrease') or \
               (limit_type == 'max' and direction.lower() == 'increase'):
                violations.append(f"{parameter} is already at {limit_type} limit per driver constraints")

        cannot_adjust = constraints.get('cannot_adjust', [])
        if parameter in cannot_adjust:
            violations.append(f"{parameter} cannot be adjusted per driver constraints")

        already_tried = constraints.get('already_tried', [])
        if parameter in already_tried:
            warnings.append(f"{parameter} was already tried in previous sessions")

    is_valid = len(violations) == 0

    result = {
        "is_valid": is_valid,
        "violations": violations,
        "warnings": warnings,
        "nascar_manual_limits": manual_limits,
        "proposed_value": proposed_value,
        "current_value": current_value,
        "within_typical_range": True  # Default
    }

    # Calculate margins if we have limits and proposed value
    if manual_limits and proposed_value is not None:
        result["margin_to_limits"] = {
            "min": proposed_value - manual_limits['min'],
            "max": manual_limits['max'] - proposed_value,
            "unit": manual_limits.get('unit', 'units')
        }

        # Check if within typical range
        if manual_limits.get('typical_min') and manual_limits.get('typical_max'):
            result["within_typical_range"] = (
                manual_limits['typical_min'] <= proposed_value <= manual_limits['typical_max']
            )

    return result
```

---

### 3. Update `race_engineer/agents.py` - Add Deduplication

**Update imports (add at top of file)**:

```python
from race_engineer.recommendation_deduplicator import check_duplicate, filter_unique
```

**Update `setup_engineer_node` function (around line 295)**:

REPLACE the entire function with:

```python
def setup_engineer_node(state: RaceEngineerState) -> Dict[str, Any]:
    """
    Setup Engineer agent generates specific recommendations.

    NOW WITH DEDUPLICATION - Never suggests the same thing twice!
    """
    print("\n" + "="*70)
    print("🔧 SETUP ENGINEER: Generating recommendations")
    print("="*70)

    llm = create_llm(temperature=0.3)

    # Bind tools
    tools = [check_constraints, validate_physics, visualize_impacts]
    llm_with_tools = llm.bind_tools(tools)

    # CHECK FOR PREVIOUS RECOMMENDATIONS - Critical for avoiding duplicates!
    previous_recs = state.get('previous_recommendations', [])
    already_recommended_params = {rec.get('parameter') for rec in previous_recs}

    print(f"\n📋 Previous recommendations: {len(previous_recs)}")
    if previous_recs:
        print("   Already recommended:")
        for rec in previous_recs:
            print(f"      • {rec.get('parameter')}: {rec.get('direction')} by {rec.get('magnitude')}")

    # Build comprehensive context with EXPLICIT deduplication instruction
    task_parts = []
    task_parts.append(f"Driver feedback: {state['driver_feedback']}")

    # CRITICAL: Tell agent what's already been suggested
    if previous_recs:
        task_parts.append("\n⚠️  PREVIOUSLY RECOMMENDED (DO NOT REPEAT THESE):")
        for rec in previous_recs:
            task_parts.append(
                f"  ❌ {rec['parameter']}: {rec['direction']} by {rec['magnitude']} {rec.get('magnitude_unit', '')}"
            )
        task_parts.append("\n✅ You MUST recommend DIFFERENT parameters or approaches!")

    # Statistical analysis context
    if state.get('statistical_analysis'):
        stats = state['statistical_analysis']
        task_parts.append(f"\nStatistical Analysis ({stats.get('method', 'unknown')}):")

        # Show impacts but filter out already recommended
        all_impacts = stats.get('correlations') or stats.get('coefficients', {})
        if all_impacts:
            sorted_impacts = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

            task_parts.append("  Available parameters (sorted by impact):")
            count = 0
            for param, impact in sorted_impacts:
                if param not in already_recommended_params:
                    direction_indicator = "↓" if impact > 0 else "↑"
                    task_parts.append(f"    {direction_indicator} {param}: {impact:+.3f}")
                    count += 1
                    if count >= 5:  # Show top 5 available
                        break

            if count == 0:
                task_parts.append("    ⚠️  All high-impact parameters already recommended!")

    # Knowledge context
    if state.get('knowledge_insights'):
        task_parts.append("\n✅ Setup knowledge consulted")

    # Instructions
    task_parts.append("\n📝 TASK:")
    task_parts.append("1. Generate setup recommendations for PARAMETERS NOT YET RECOMMENDED")
    task_parts.append("2. Use check_constraints to validate each recommendation")
    task_parts.append("3. Provide specific magnitudes with units (PSI, lb/in, %, inches)")
    task_parts.append("4. If all good parameters are exhausted, say so clearly")

    if state.get('driver_constraints'):
        task_parts.append(f"\nDriver Constraints: {state['driver_constraints']}")

    messages = [
        SystemMessage(content=get_setup_engineer_prompt()),
        HumanMessage(content="\n".join(task_parts))
    ]

    response = llm_with_tools.invoke(messages)

    new_messages = [response]
    updates = {"agents_consulted": state['agents_consulted'] + ['setup_engineer']}

    # Process tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\n🔧 Calling {len(response.tool_calls)} tool(s)...")

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            print(f"   → {tool_name}({list(tool_args.keys())})")

            tool_result = _execute_tool(tool_name, tool_args, tools)

            if tool_name == 'visualize_impacts' and 'error' not in str(tool_result):
                updates['generated_visualizations'] = state.get('generated_visualizations', []) + [tool_result]
                updates['tools_called'] = state['tools_called'] + ['visualize_impacts']

            from langchain_core.messages import ToolMessage
            new_messages.append(ToolMessage(
                content=json.dumps(tool_result, default=str),
                tool_call_id=tool_call['id']
            ))

        # Get final recommendations after tools
        summary_response = llm.invoke(messages + new_messages)
        new_messages.append(summary_response)
        print(f"\n📝 Recommendations: {summary_response.content[:300]}...")

        # Parse recommendations from response
        new_recommendations = _parse_recommendations(summary_response.content, state)

        if new_recommendations:
            # DEDUPLICATE before adding to state!
            dedup_result = filter_unique(new_recommendations, previous_recs)

            unique_recs = dedup_result['unique']
            duplicate_recs = dedup_result['duplicates']

            print(f"\n✅ New recommendations: {len(unique_recs)}")
            print(f"🚫 Filtered duplicates: {len(duplicate_recs)}")

            if duplicate_recs:
                print("   Duplicates filtered:")
                for dup in duplicate_recs:
                    print(f"      • {dup.get('parameter')}: {dup.get('direction')}")

            # Add unique recommendations to previous
            all_previous = previous_recs.copy()
            for rec in unique_recs:
                rec['iteration_made'] = state['iteration']
                rec['agent_source'] = 'setup_engineer'
                all_previous.append(rec)

                # Update parameter history
                param = rec['parameter']
                param_history = state.get('parameter_adjustment_history', {})
                if param not in param_history:
                    param_history[param] = []

                param_history[param].append({
                    'iteration': state['iteration'],
                    'direction': rec['direction'],
                    'magnitude': rec['magnitude'],
                    'result': 'proposed'
                })

                updates['parameter_adjustment_history'] = param_history

            # Update state
            updates['previous_recommendations'] = all_previous
            updates['candidate_recommendations'] = unique_recs

            # Update stats
            stats = state.get('recommendation_stats', {
                'total_proposed': 0,
                'unique_accepted': 0,
                'duplicates_filtered': 0,
                'constraint_violations_caught': 0,
                'parameters_touched': []
            })

            stats['total_proposed'] += len(new_recommendations)
            stats['unique_accepted'] += len(unique_recs)
            stats['duplicates_filtered'] += len(duplicate_recs)
            stats['parameters_touched'] = list(set(stats.get('parameters_touched', []) + [r['parameter'] for r in unique_recs]))

            updates['recommendation_stats'] = stats

            # Set final recommendation
            if unique_recs:
                updates['final_recommendation'] = {
                    "primary": unique_recs[0] if unique_recs else None,
                    "secondary": unique_recs[1:] if len(unique_recs) > 1 else [],
                    "summary": summary_response.content,
                    "num_unique": len(unique_recs),
                    "num_filtered": len(duplicate_recs)
                }

    updates['messages'] = new_messages

    return updates
```

---

### 4. Update `race_engineer/prompts.py` - NASCAR Knowledge

**Update KNOWLEDGE_EXPERT_SYSTEM_PROMPT (around line 92)**:

REPLACE with:

```python
KNOWLEDGE_EXPERT_SYSTEM_PROMPT = """You are a NASCAR Setup Knowledge Expert with deep understanding of oval racing physics and setup theory.

YOUR EXPERTISE:
- NASCAR Truck Series setup principles (from official NASCAR Trucks Manual V6)
- Bristol Motor Speedway specific knowledge (short oval, high banking)
- Historical pattern recognition
- Setup balance and vehicle dynamics

AVAILABLE TOOLS:
- query_setup_manual: Search NASCAR Trucks Manual V6 for handling issues and parameter guidance
- search_history: Find similar historical sessions with same complaints

KEY KNOWLEDGE FROM NASCAR TRUCKS MANUAL V6:

TIRE PRESSURE (25-35 PSI, typical 28-30 PSI):
- Lower pressure = more contact patch = more mechanical grip
- Higher pressure = less heat buildup, better for high speeds
- RF tire most heavily loaded on left-turn ovals - critical for turn entry
- RR tire critical for exit traction

CROSS WEIGHT (50-56%, typical 52-54%):
- Percentage of weight on LR and RF tires
- ONE OF THE MOST INFLUENTIAL SETTINGS
- Higher = stabilizes entry, helps drive-off, can cause mid-corner understeer
- Lower = frees rotation, can cause exit instability
- ONLY adjustment needed for temperature changes!

TRACK BAR HEIGHT (typical 7-12 inches):
- Controls rear roll center
- Higher = more rear roll stiffness = OVERSTEER
- Lower = more traction = UNDERSTEER
- Positive rake (right higher) adds exit oversteer

SPRINGS:
- Front: Pigtail coil-bind springs (200→selected rate when bound)
- LF: 375-450 lb/in typical
- RF: 400-500 lb/in typical (stiffer, most loaded)
- LR: 300-375 lb/in (softer allows roll, frees car)
- RR: 800-1100 lb/in (very stiff for height control)

BRISTOL SPECIFIC:
- Short oval (0.533 miles)
- High banking (24-28°) loads right-side heavily
- Tight radius requires strong turn-in
- Lower RR pressure helps exit traction

HANDLING DIAGNOSIS:

OVERSTEER (Loose, rear slides):
→ Reduce RR/LR tire pressure (1-2 PSI)
→ Lower track bar height (0.25-0.5")
→ Increase cross weight (0.5-1.0%)
→ Soften LR spring (25-50 lb/in)

UNDERSTEER (Tight, won't turn):
→ Reduce LF/RF tire pressure (1-2 PSI)
→ Decrease cross weight (0.5-1.0%)
→ Soften front springs
→ Stiffen LR spring

CRITICAL: Use query_setup_manual tool to get specific guidance from NASCAR manual!

YOUR WORKFLOW:
1. Understand the driver's handling complaint
2. Use query_setup_manual to get NASCAR Trucks Manual guidance
3. Use search_history to find similar past sessions (optional)
4. Provide context that helps interpret the statistical data
5. Explain the physics behind recommendations

RESPONSE STYLE:
- Reference NASCAR Trucks Manual V6 when providing guidance
- Provide specific parameter ranges and typical values
- Explain WHY each change helps (physics/mechanics)
- Note when data contradicts traditional wisdom
- When done: "Knowledge context provided. Key principles: [summary]"
"""
```

**Update SETUP_ENGINEER_SYSTEM_PROMPT (around line 136)**:

ADD after line 143:

```python
CRITICAL - DEDUPLICATION:
Before making ANY recommendation:
1. Check what parameters have ALREADY been recommended
2. NEVER suggest the same parameter twice
3. If all good parameters exhausted, say so explicitly
4. Find alternative parameters or approaches

```

---

### 5. Update `demo.py` - Polish Output

**Update `format_output` function (around line 100)**:

REPLACE entire function with version that shows constraints, margins, and filtered duplicates:

```python
def format_output(state: dict, df: pd.DataFrame, request: AnalysisRequest,
                  using_real_data: bool, verbose: bool = False) -> str:
    """Enhanced output with NASCAR manual constraints and deduplication info"""

    output_lines = []

    # Header
    output_lines.append("=" * 70)
    output_lines.append("AI RACE ENGINEER - NASCAR Trucks Setup Analysis")
    output_lines.append("=" * 70)
    output_lines.append("")

    # Driver feedback summary (if present)
    if request.driver_feedback:
        fb = request.driver_feedback
        output_lines.append(f"🎧 Driver Feedback: {fb.complaint.replace('_', ' ').title()}")
        output_lines.append(f"   {fb.description}")
        output_lines.append("")

    # Primary recommendation with full context
    if state.get('final_recommendation'):
        final_rec = state['final_recommendation']
        primary = final_rec.get('primary')

        if primary:
            output_lines.append("💡 PRIMARY RECOMMENDATION:")
            output_lines.append(f"   {primary['parameter'].replace('_', ' ').title()}")
            output_lines.append(f"   {primary['direction'].title()} by {primary['magnitude']} {primary.get('magnitude_unit', 'units')}")
            output_lines.append("")

            # Show NASCAR manual constraint context if available
            if 'constraint_validation' in primary:
                validation = primary['constraint_validation']
                limits = validation.get('nascar_manual_limits', {})

                if limits:
                    output_lines.append(f"   NASCAR Manual Range: {limits['min']}-{limits['max']} {limits.get('unit', '')}")

                    if validation.get('proposed_value'):
                        current = validation.get('current_value', '?')
                        proposed = validation['proposed_value']
                        output_lines.append(f"   Current:  {current}")
                        output_lines.append(f"   Proposed: {proposed}")

                        margins = validation.get('margin_to_limits', {})
                        if margins:
                            output_lines.append(
                                f"   Margin:   {margins.get('min', 0):.1f} from min | "
                                f"{margins.get('max', 0):.1f} from max"
                            )

                        if not validation.get('within_typical_range', True):
                            output_lines.append(f"   ⚠️  Outside typical range")

                    output_lines.append(f"   Source:   {limits.get('source', 'NASCAR Trucks Manual V6')}")
                    output_lines.append("")

            # Expected impact
            output_lines.append(f"   Impact:     {primary.get('expected_impact', 'Correlation detected')}")
            output_lines.append(f"   Confidence: {int(primary.get('confidence', 0.5) * 100)}%")
            output_lines.append("")

            # Rationale
            rationale = primary.get('rationale', 'Statistical correlation with lap time')
            output_lines.append(f"   Why: {rationale}")
            output_lines.append("")

        # Show secondary recommendations if verbose
        if verbose and final_rec.get('secondary'):
            output_lines.append("📋 SECONDARY RECOMMENDATIONS:")
            for i, rec in enumerate(final_rec['secondary'][:3], 1):
                output_lines.append(
                    f"   {i}. {rec['parameter'].replace('_', ' ').title()}: "
                    f"{rec['direction']} by {rec['magnitude']} {rec.get('magnitude_unit', '')}"
                )
            output_lines.append("")

        # Show deduplication stats
        if final_rec.get('num_filtered', 0) > 0:
            output_lines.append(
                f"ℹ️  Note: Filtered {final_rec['num_filtered']} duplicate recommendation(s)"
            )
            output_lines.append("")

    # Top impactful parameters (concise) or Top 5 (verbose)
    analysis = state.get('statistical_analysis', {})
    if analysis:
        all_impacts = analysis.get('correlations') or analysis.get('coefficients', {})
        if all_impacts:
            sorted_impacts = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
            num_to_show = 5 if verbose else 3

            output_lines.append("📊 PARAMETER IMPACTS:")
            for param, impact in sorted_impacts[:num_to_show]:
                direction = "↓" if impact > 0 else "↑"
                action = "Reduce" if impact > 0 else "Increase"
                output_lines.append(f"   {direction} {param:25s}  {action:8s}  ({impact:+.3f})")
            output_lines.append("")

    # Performance summary
    best_time = float(df['fastest_time'].min())
    baseline_time = float(df['fastest_time'].max())
    improvement = baseline_time - best_time

    output_lines.append("⚡ PERFORMANCE:")
    output_lines.append(f"   Current Best:  {best_time:.3f}s")
    output_lines.append(f"   Potential:     {baseline_time - improvement - 0.050:.3f}s  (↓{improvement + 0.050:.3f}s)")
    output_lines.append("")

    # Recommendation stats
    stats = state.get('recommendation_stats', {})
    if verbose and stats.get('total_proposed', 0) > 0:
        output_lines.append("📈 SESSION STATISTICS:")
        output_lines.append(f"   Total recommendations proposed: {stats.get('total_proposed', 0)}")
        output_lines.append(f"   Unique accepted:                {stats.get('unique_accepted', 0)}")
        output_lines.append(f"   Duplicates filtered:            {stats.get('duplicates_filtered', 0)}")
        output_lines.append(f"   Parameters touched:             {len(stats.get('parameters_touched', []))}")
        output_lines.append("")

    # Data source if verbose
    if verbose:
        data_source = "Real telemetry" if using_real_data else "Mock demo data"
        output_lines.append(f"📁 Data: {data_source} ({len(df)} sessions)")
        output_lines.append("")

    output_lines.append("=" * 70)

    return "\n".join(output_lines)
```

---

## Quick Implementation Commands

```bash
# 1. Update state.py - manually add fields from section 1

# 2. Update tools.py - manually replace functions from section 2

# 3. Update agents.py - manually replace setup_engineer_node from section 3

# 4. Update prompts.py - manually update prompts from section 4

# 5. Update demo.py - manually update format_output from section 5

# 6. Test the improvements
python demo.py "The car is really loose in turns 1 and 2, rear end wants to come around"

# Should see:
# - NASCAR manual constraints shown
# - No duplicate recommendations
# - Professional output format
```

---

## Success Criteria

✅ **No Duplicates**: Run demo twice with same feedback - should get different parameters
✅ **NASCAR Manual**: See "NASCAR Manual Range: X-Y PSI" in output
✅ **Constraint Validation**: Recommendations within valid ranges
✅ **Professional Output**: Clean, informative, with margins and context
✅ **Deduplication Stats**: Shows "Filtered X duplicates" when applicable

---

## Testing Checklist

- [ ] Run demo with oversteer complaint
- [ ] Verify no duplicate recommendations
- [ ] Check NASCAR manual limits appear in output
- [ ] Verify constraint violations are caught
- [ ] Test with extreme values (should be rejected)
- [ ] Run multiple iterations (should get different parameters)
- [ ] Check verbose mode shows all details
- [ ] Verify deduplication stats are accurate

---

END OF IMPLEMENTATION SCRIPT
