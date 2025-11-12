# Agent Polish Implementation Instructions

## Context
You have a NASCAR race engineer agentic system built with LangGraph. Goal: Upgrade from Level 2 to Level 2.5 agent maturity by implementing explicit Think-Act-Observe loops, LM-as-judge evaluation, and improved supervisor synthesis. Target: 4-6 hours of focused implementation for interview-ready MVP.

## Phase 1: Explicit Think-Act-Observe Loop (Priority 1)

### File: `race_engineer/agents.py`

**Modify `setup_engineer_node` function (currently ~line 295)**

REPLACE the current single-pass tool calling with this iterative reflection loop:

```python
def setup_engineer_node(state: RaceEngineerState) -> Dict[str, Any]:
    """Setup Engineer with explicit Think-Act-Observe loop"""
    
    print("\n" + "="*70)
    print("🔧 SETUP ENGINEER: Starting Think-Act-Observe Loop")
    print("="*70)
    
    llm = create_llm(temperature=0.3)
    max_cycles = 3
    cycle = 0
    
    # Prepare context once
    context = _build_engineer_context(state)
    
    while cycle < max_cycles:
        cycle += 1
        print(f"\n--- Cycle {cycle}/{max_cycles} ---")
        
        # === THINK PHASE ===
        print("💭 THINK: Planning approach...")
        think_prompt = f"""Given this context:
{context}

Previous cycles: {cycle - 1}

THINK: What's your strategy?
1. What tools do you need to call?
2. What specific questions need answers?
3. What's your confidence in the current data?

Respond with your reasoning and planned tool calls."""

        thinking = llm.invoke([
            SystemMessage(content=get_setup_engineer_prompt()),
            HumanMessage(content=think_prompt)
        ])
        
        print(f"Strategy: {thinking.content[:200]}...")
        
        # === ACT PHASE ===
        print("\n🔨 ACT: Executing tools...")
        llm_with_tools = llm.bind_tools([check_constraints, validate_physics])
        
        action_prompt = f"""Based on your strategy:
{thinking.content}

Now call the appropriate tools to gather information.
Context: {context}"""

        action_response = llm_with_tools.invoke([
            SystemMessage(content=get_setup_engineer_prompt()),
            HumanMessage(content=action_prompt)
        ])
        
        tool_results = []
        if hasattr(action_response, 'tool_calls') and action_response.tool_calls:
            for tc in action_response.tool_calls:
                result = _execute_tool(tc['name'], tc['args'], [check_constraints, validate_physics])
                tool_results.append({tc['name']: result})
                print(f"  ✓ {tc['name']}: {str(result)[:100]}...")
        
        # === OBSERVE PHASE ===
        print("\n👁️  OBSERVE: Reflecting on results...")
        observe_prompt = f"""You planned: {thinking.content}
You executed tools and got: {json.dumps(tool_results, default=str)}

OBSERVE: 
1. Did you get the information you needed?
2. What's your confidence level now (0-1)?
3. Do you need another cycle or are you ready to recommend?

Respond with JSON:
{{"confidence": 0.8, "ready": true, "gaps": "string or null", "preliminary_rec": "..."}}"""

        observation = llm.invoke([
            SystemMessage(content="You are evaluating your own work."),
            HumanMessage(content=observe_prompt)
        ])
        
        print(f"Observation: {observation.content[:200]}...")
        
        # Parse observation
        try:
            obs_data = json.loads(observation.content.strip().replace("```json", "").replace("```", ""))
            confidence = obs_data.get('confidence', 0.5)
            ready = obs_data.get('ready', False)
            
            print(f"  Confidence: {confidence:.1%}")
            
            if ready and confidence > 0.7:
                print("  ✓ Ready to recommend")
                break
            elif cycle == max_cycles:
                print("  ⚠ Max cycles reached, proceeding with current confidence")
                break
            else:
                print(f"  ↻ Not ready (confidence: {confidence:.1%}), continuing...")
                context += f"\n\nCycle {cycle} findings: {obs_data.get('gaps', 'Refining approach')}"
                
        except:
            print("  ⚠ Could not parse observation, proceeding")
            break
    
    # Generate final recommendation
    print("\n📝 Generating final recommendation...")
    final_rec = _generate_final_recommendation(state, tool_results, llm)
    
    return {
        "agents_consulted": state['agents_consulted'] + ['setup_engineer'],
        "final_recommendation": final_rec,
        "candidate_recommendations": final_rec.get('recommendations', []),
        "messages": []  # Add message tracking if needed
    }

def _build_engineer_context(state):
    """Extract relevant context for engineer"""
    parts = [f"Driver feedback: {state['driver_feedback']}"]
    
    if state.get('statistical_analysis'):
        stats = state['statistical_analysis']
        parts.append(f"Top correlation: {stats.get('top_parameter')} ({stats.get('top_correlation'):.3f})")
    
    if state.get('knowledge_insights'):
        parts.append("NASCAR manual guidance: Available")
    
    previous = state.get('previous_recommendations', [])
    if previous:
        parts.append(f"Already tried: {[p['parameter'] for p in previous]}")
    
    return "\n".join(parts)

def _generate_final_recommendation(state, tool_results, llm):
    """Generate structured final recommendation"""
    
    # Use statistical analysis + tool results to create recommendation
    stats = state.get('statistical_analysis', {})
    all_impacts = stats.get('correlations') or stats.get('coefficients', {})
    
    if not all_impacts:
        return {"primary": None, "summary": "Insufficient data for recommendation"}
    
    # Get top parameter
    sorted_params = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    top_param, top_impact = sorted_params[0]
    
    direction = "decrease" if top_impact > 0 else "increase"
    magnitude, unit = _estimate_magnitude(top_param)
    
    rec = {
        "primary": {
            "parameter": top_param,
            "direction": direction,
            "magnitude": magnitude,
            "magnitude_unit": unit,
            "confidence": 0.8 if abs(top_impact) > 0.3 else 0.6,
            "rationale": f"Statistical analysis shows {top_impact:+.3f} correlation with lap time",
            "tool_validations": tool_results
        },
        "recommendations": [{
            "parameter": top_param,
            "direction": direction,
            "magnitude": magnitude,
            "magnitude_unit": unit
        }],
        "summary": f"Primary: {direction} {top_param} by {magnitude} {unit}"
    }
    
    return rec
```

**Add helper at end of file:**

```python
def _estimate_magnitude(parameter: str) -> tuple:
    """Estimate appropriate magnitude for a parameter"""
    if 'psi' in parameter.lower() or 'tire' in parameter.lower():
        return 1.5, "PSI"
    elif 'spring' in parameter.lower():
        return 25.0, "lb/in"
    elif 'cross_weight' in parameter.lower():
        return 0.5, "%"
    elif 'track_bar' in parameter.lower():
        return 0.25, "inches"
    else:
        return 1.0, "units"
```

---

## Phase 2: LM-as-Judge Evaluation (Priority 2)

### File: `race_engineer/tools.py`

**Add new tool after `validate_physics` (around line 680):**

```python
@tool
def evaluate_recommendation_quality(
    recommendation: Dict[str, Any],
    driver_feedback: str,
    statistical_support: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    LM-as-judge evaluation of recommendation quality.
    
    Implements Google AgentOps principle: "Quality Instead of Pass/Fail"
    
    Evaluates on 4 dimensions:
    - Relevance: Does it address driver's complaint?
    - Confidence: Is statistical support strong?
    - Safety: Are constraints respected?
    - Clarity: Is guidance specific and actionable?
    
    Returns quality scores and pass/fail decision.
    """
    
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage
    
    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.1)
    
    # Extract key info
    param = recommendation.get('parameter', 'unknown')
    direction = recommendation.get('direction', 'unknown')
    magnitude = recommendation.get('magnitude', 0)
    unit = recommendation.get('magnitude_unit', '')
    
    # Build evaluation prompt
    eval_prompt = f"""You are a NASCAR setup quality judge. Evaluate this recommendation:

DRIVER COMPLAINT: {driver_feedback}

RECOMMENDATION: {direction.title()} {param} by {magnitude} {unit}

STATISTICAL SUPPORT:
- Method: {statistical_support.get('method', 'unknown')}
- Top correlation: {statistical_support.get('top_correlation', 'N/A')}
- Significant params: {statistical_support.get('significant_params', [])}

CONSTRAINTS: {json.dumps(constraints) if constraints else 'None provided'}

Rate on scale 0-10:
1. RELEVANCE: Does this address "{driver_feedback}"?
2. CONFIDENCE: Is the statistical support strong enough?
3. SAFETY: Does it respect limits and constraints?
4. CLARITY: Is it specific and actionable?

Respond ONLY with valid JSON (no markdown):
{{
  "relevance": 8,
  "confidence": 7,
  "safety": 10,
  "clarity": 9,
  "overall_score": 8.5,
  "pass": true,
  "reasoning": "Strong statistical support (correlation -0.42) directly addresses oversteer complaint. Within NASCAR manual limits."
}}"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are an impartial quality judge for NASCAR setup recommendations."),
            HumanMessage(content=eval_prompt)
        ])
        
        # Parse JSON from response
        content = response.content.strip()
        # Remove markdown code blocks if present
        content = content.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(content)
        
        # Ensure all required fields
        result.setdefault('relevance', 5)
        result.setdefault('confidence', 5)
        result.setdefault('safety', 5)
        result.setdefault('clarity', 5)
        result.setdefault('overall_score', 5.0)
        result.setdefault('pass', result['overall_score'] >= 7.0)
        result.setdefault('reasoning', 'No reasoning provided')
        
        return {
            "evaluation": result,
            "recommendation_validated": result['pass'],
            "quality_gate": "passed" if result['pass'] else "failed",
            "improvement_areas": [
                k for k, v in result.items() 
                if isinstance(v, (int, float)) and v < 7
            ]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "evaluation": None,
            "recommendation_validated": False,
            "quality_gate": "error"
        }
```

**Update `ALL_TOOLS` list at bottom of file:**

```python
ALL_TOOLS = [
    load_telemetry,
    inspect_quality,
    clean_data,
    select_features,
    correlation_analysis,
    regression_analysis,
    query_setup_manual,
    search_history,
    check_constraints,
    validate_physics,
    visualize_impacts,
    save_session,
    evaluate_recommendation_quality  # ADD THIS
]
```

### File: `race_engineer/agents.py`

**Modify `supervisor_node` to call judge before COMPLETE (around line 80):**

```python
def supervisor_node(state: RaceEngineerState) -> Dict[str, Any]:
    """Supervisor with LM-as-judge quality gate"""
    
    iteration = state['iteration'] + 1
    agents_consulted = state.get('agents_consulted', [])
    
    # If setup_engineer ran, evaluate before completing
    if 'setup_engineer' in agents_consulted:
        final_rec = state.get('final_recommendation')
        
        if final_rec and final_rec.get('primary'):
            print("\n🎯 QUALITY GATE: Evaluating recommendation...")
            
            from race_engineer.tools import evaluate_recommendation_quality
            
            evaluation = evaluate_recommendation_quality.invoke({
                'recommendation': final_rec['primary'],
                'driver_feedback': state['driver_feedback'],
                'statistical_support': state.get('statistical_analysis', {}),
                'constraints': state.get('driver_constraints')
            })
            
            print(f"  Quality Score: {evaluation['evaluation']['overall_score']:.1f}/10")
            print(f"  Status: {evaluation['quality_gate'].upper()}")
            
            if not evaluation['recommendation_validated']:
                print(f"  ⚠ FAILED: {evaluation['evaluation']['reasoning']}")
                print(f"  Improvement needed: {evaluation['improvement_areas']}")
                
                # Could route back to setup_engineer here, but for MVP just flag
                state['warnings'] = state.get('warnings', []) + [
                    f"Quality gate failed: {evaluation['evaluation']['reasoning']}"
                ]
            else:
                print(f"  ✓ PASSED: {evaluation['evaluation']['reasoning']}")
            
            # Add evaluation to state
            state['recommendation_evaluation'] = evaluation
        
        return {
            "messages": [],
            "next_agent": "COMPLETE",
            "iteration": iteration
        }
    
    # ... rest of existing supervisor logic ...
```

---

## Phase 3: Supervisor Synthesis (Priority 3)

### File: `race_engineer/agents.py`

**Add synthesis step to `supervisor_node` BEFORE routing to COMPLETE:**

```python
def supervisor_node(state: RaceEngineerState) -> Dict[str, Any]:
    """Supervisor with multi-agent synthesis"""
    
    iteration = state['iteration'] + 1
    agents_consulted = state.get('agents_consulted', [])
    
    # SYNTHESIS: If we have insights from multiple agents, synthesize
    if len(agents_consulted) >= 2 and 'data_analyst' in agents_consulted:
        
        print("\n🔄 SYNTHESIS: Reconciling multi-agent insights...")
        
        insights = {
            'data': state.get('statistical_analysis'),
            'knowledge': state.get('knowledge_insights'),
            'recommendations': state.get('candidate_recommendations')
        }
        
        # Check for conflicts
        conflicts = _detect_conflicts(insights)
        
        if conflicts:
            print(f"  ⚠ Conflicts detected: {conflicts}")
            
            # Synthesize with LLM
            llm = create_llm(temperature=0.2)
            
            synthesis_prompt = f"""You have insights from multiple specialist agents:

DATA ANALYST: {json.dumps(insights['data'], default=str)[:500]}
KNOWLEDGE EXPERT: {json.dumps(insights['knowledge'], default=str)[:500]}
RECOMMENDATIONS: {json.dumps(insights['recommendations'], default=str)[:500]}

CONFLICTS DETECTED: {conflicts}

TASK: Provide a synthesized recommendation that:
1. Resolves conflicts with clear reasoning
2. Prioritizes based on confidence and safety
3. Acknowledges uncertainties

Respond with: "SYNTHESIS: [your unified recommendation]" """

            synthesis = llm.invoke([
                SystemMessage(content=get_supervisor_prompt()),
                HumanMessage(content=synthesis_prompt)
            ])
            
            print(f"  ✓ Synthesis: {synthesis.content[:200]}...")
            
            # Update state with synthesis
            state['supervisor_synthesis'] = synthesis.content
        else:
            print("  ✓ No conflicts - insights align")
    
    # ... continue with existing routing logic ...
```

**Add helper function:**

```python
def _detect_conflicts(insights: Dict) -> List[str]:
    """Detect conflicts between agent insights"""
    conflicts = []
    
    # Example: Data says X but knowledge says Y
    data_top = insights.get('data', {}).get('top_parameter')
    
    if insights.get('knowledge'):
        knowledge_params = insights['knowledge'].get('parameter_guidance', {})
        if data_top and data_top not in knowledge_params:
            conflicts.append(f"Data suggests {data_top} but knowledge has no guidance on it")
    
    # Check if recommendations conflict with constraints
    recs = insights.get('recommendations', [])
    for rec in recs:
        if rec.get('constraint_violations'):
            conflicts.append(f"{rec['parameter']}: Violates constraints")
    
    return conflicts
```

---

## Phase 4: Metrics & Instrumentation (Priority 4)

### File: `race_engineer/state.py`

**Add to `RaceEngineerState` class (around line 150):**

```python
    # ===== METRICS & INSTRUMENTATION =====
    agent_metrics: Dict[str, Any]
    """Performance metrics per agent:
       {
           'data_analyst': {
               'duration_seconds': 2.3,
               'tool_calls': 5,
               'tokens_used': 1500,
               'cost_estimate': 0.002
           }
       }
    """
    
    total_cost_estimate: float
    """Cumulative cost estimate for this session (USD)"""
```

**Update `create_initial_state` to initialize (around line 240):**

```python
        # Metadata
        "session_id": session_id,
        "session_timestamp": datetime.utcnow().isoformat(),
        
        # ADD THESE:
        "agent_metrics": {},
        "total_cost_estimate": 0.0
```

### File: `race_engineer/agents.py`

**Add metric tracking wrapper (add at top of file after imports):**

```python
from time import time
from functools import wraps

def track_agent_metrics(agent_name: str):
    """Decorator to track agent performance metrics"""
    def decorator(agent_func):
        @wraps(agent_func)
        def wrapper(state: RaceEngineerState):
            start_time = time()
            
            # Call agent
            result = agent_func(state)
            
            end_time = time()
            duration = end_time - start_time
            
            # Track metrics
            metrics = state.get('agent_metrics', {})
            metrics[agent_name] = {
                'duration_seconds': round(duration, 2),
                'tool_calls': len(state.get('tools_called', [])) - len(metrics),  # Incremental
                'timestamp': datetime.now().isoformat()
            }
            
            # Estimate cost (Haiku: $0.25 per 1M input tokens, ~1K tokens per call)
            estimated_tokens = 1500  # Conservative estimate
            cost_per_token = 0.00000025  # $0.25 / 1M
            cost = estimated_tokens * cost_per_token
            
            metrics[agent_name]['cost_estimate'] = round(cost, 4)
            
            result['agent_metrics'] = metrics
            result['total_cost_estimate'] = state.get('total_cost_estimate', 0) + cost
            
            print(f"\n📊 Metrics: {agent_name} - {duration:.2f}s, ~${cost:.4f}")
            
            return result
        return wrapper
    return decorator
```

**Apply decorator to agents:**

```python
@track_agent_metrics('data_analyst')
def data_analyst_node(state: RaceEngineerState) -> Dict[str, Any]:
    # ... existing code ...

@track_agent_metrics('knowledge_expert')
def knowledge_expert_node(state: RaceEngineerState) -> Dict[str, Any]:
    # ... existing code ...

@track_agent_metrics('setup_engineer')
def setup_engineer_node(state: RaceEngineerState) -> Dict[str, Any]:
    # ... existing code ...
```

---

## Phase 5: Enhanced Demo Output

### File: `main.py`

**Update `display_results` function to show new features (around line 100):**

```python
def display_results(state: dict, verbose: bool = False):
    """Display results with Think-Act-Observe and quality metrics"""
    
    print("\n" + "="*70)
    print("📋 FINAL RESULTS")
    print("="*70)
    
    # Show Think-Act-Observe cycles if verbose
    if verbose and state.get('messages'):
        print("\n🔄 AGENT WORKFLOW:")
        for msg in state['messages'][-5:]:  # Last 5 messages
            if hasattr(msg, 'content'):
                preview = msg.content[:100].replace('\n', ' ')
                print(f"  • {msg.__class__.__name__}: {preview}...")
    
    # Quality gate results
    if state.get('recommendation_evaluation'):
        eval_data = state['recommendation_evaluation']['evaluation']
        print(f"\n🎯 QUALITY EVALUATION:")
        print(f"   Overall Score: {eval_data['overall_score']:.1f}/10")
        print(f"   Relevance:  {eval_data['relevance']}/10")
        print(f"   Confidence: {eval_data['confidence']}/10")
        print(f"   Safety:     {eval_data['safety']}/10")
        print(f"   Status:     {'✓ PASSED' if eval_data['pass'] else '✗ FAILED'}")
        if not eval_data['pass']:
            print(f"   Reason: {eval_data['reasoning']}")
    
    # Primary recommendation
    final_rec = state.get('final_recommendation')
    if final_rec and final_rec.get('primary'):
        primary = final_rec['primary']
        print(f"\n💡 PRIMARY RECOMMENDATION:")
        print(f"   {primary['parameter'].replace('_', ' ').title()}")
        print(f"   {primary['direction'].title()} by {primary['magnitude']} {primary['magnitude_unit']}")
        print(f"   Confidence: {int(primary.get('confidence', 0.8) * 100)}%")
        print(f"   Rationale: {primary.get('rationale', 'Statistical correlation')}")
    else:
        print("\n⚠️  WARNING: No recommendations generated")
    
    # Agent metrics
    if verbose and state.get('agent_metrics'):
        print(f"\n📊 AGENT PERFORMANCE:")
        total_time = 0
        for agent, metrics in state['agent_metrics'].items():
            print(f"   {agent:20s}: {metrics['duration_seconds']:>5.2f}s, ${metrics['cost_estimate']:.4f}")
            total_time += metrics['duration_seconds']
        
        total_cost = state.get('total_cost_estimate', 0)
        print(f"   {'TOTAL':20s}: {total_time:>5.2f}s, ${total_cost:.4f}")
    
    # Synthesis if available
    if state.get('supervisor_synthesis'):
        print(f"\n🔄 SUPERVISOR SYNTHESIS:")
        print(f"   {state['supervisor_synthesis'][:200]}...")
    
    print("\n" + "="*70)
```

---

## Testing Instructions

**Run with verbose mode to see all new features:**

```bash
python demo.py "Car is loose on corner exit" --verbose
```

**Expected output should show:**
1. Think-Act-Observe cycles with iteration counts
2. LM-as-judge quality scores (0-10 scale)
3. Supervisor synthesis if conflicts detected
4. Agent performance metrics (time, cost)
5. Professional formatted recommendation

---

## Success Criteria

✅ **Think-Act-Observe**: Agent prints "THINK", "ACT", "OBSERVE" with confidence scores
✅ **LM-as-Judge**: Quality gate shows scores and pass/fail
✅ **Synthesis**: Supervisor reconciles conflicts explicitly
✅ **Metrics**: Every agent shows duration and cost estimate
✅ **Clean Demo**: Single command produces professional output in <10 seconds

---

## Token Optimization Note

This implementation focuses on:
- Minimal new code (leveraging existing structure)
- Reusable helpers (_estimate_magnitude, _detect_conflicts)
- Decorator pattern for metrics (DRY principle)
- Strategic print statements (visible progress without noise)

Estimated LOC: ~400 new lines across 3 files
Implementation time: 4-6 hours
