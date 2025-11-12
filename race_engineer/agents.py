"""
Agent node implementations for the Race Engineer workflow.

Each agent is a specialized node in the LangGraph that performs specific tasks.
"""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from race_engineer.state import RaceEngineerState
from race_engineer.prompts import (
    get_supervisor_prompt,
    get_data_analyst_prompt,
    get_knowledge_expert_prompt,
    get_setup_engineer_prompt
)
from race_engineer.tools import (
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
    save_session
)
from race_engineer.recommendation_deduplicator import check_duplicate, filter_unique
import re
import json
from time import time
from functools import wraps
from datetime import datetime


def create_llm(temperature: float = 0.3) -> ChatAnthropic:
    """
    Create Claude LLM instance.

    Using Haiku for cost efficiency while maintaining quality.
    """
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=temperature
    )


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

            print(f"\nMetrics: {agent_name} - {duration:.2f}s, ~${cost:.4f}")

            return result
        return wrapper
    return decorator


# ===== SUPERVISOR AGENT =====

def supervisor_node(state: RaceEngineerState) -> Dict[str, Any]:
    """
    Supervisor agent orchestrates the workflow with LM-as-judge quality gate.

    Decides which specialist agent to call next or when to complete.
    """
    iteration = state['iteration'] + 1
    max_iter = state['max_iterations']
    print(f"\n[{iteration}/{max_iter}]  SUPERVISOR: Routing decision...")

    # CRITICAL FIX: Check if we already have recommendations and should complete
    agents_consulted = state.get('agents_consulted', [])

    # SYNTHESIS: If we have insights from multiple agents, synthesize
    if len(agents_consulted) >= 2 and 'data_analyst' in agents_consulted:

        print("\nSYNTHESIS: Reconciling multi-agent insights...")

        insights = {
            'data': state.get('statistical_analysis'),
            'knowledge': state.get('knowledge_insights'),
            'recommendations': state.get('candidate_recommendations')
        }

        # Check for conflicts
        conflicts = _detect_conflicts(insights)

        if conflicts:
            print(f"  Conflicts detected: {conflicts}")

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

            print(f"  Synthesis: {synthesis.content}")

            # Update state with synthesis
            state['supervisor_synthesis'] = synthesis.content
        else:
            print("  No conflicts - insights align")

    # If setup_engineer already ran, evaluate before completing
    if 'setup_engineer' in agents_consulted:
        final_rec = state.get('final_recommendation')

        if final_rec and final_rec.get('primary'):
            print("\nQUALITY GATE: Evaluating recommendation...")

            from race_engineer.tools import evaluate_recommendation_quality

            evaluation = evaluate_recommendation_quality.invoke({
                'recommendation': final_rec['primary'],
                'driver_feedback': state['driver_feedback'],
                'statistical_support': state.get('statistical_analysis') or {},
                'constraints': state.get('driver_constraints') or {}
            })

            print(f"  Quality Score: {evaluation['evaluation']['overall_score']:.1f}/10")
            print(f"  Status: {evaluation['quality_gate'].upper()}")

            if not evaluation['recommendation_validated']:
                print(f"  FAILED: {evaluation['evaluation']['reasoning']}")
                print(f"  Improvement needed: {evaluation['improvement_areas']}")

                # Could route back to setup_engineer here, but for MVP just flag
                state['warnings'] = state.get('warnings', []) + [
                    f"Quality gate failed: {evaluation['evaluation']['reasoning']}"
                ]
            else:
                print(f"  PASSED: {evaluation['evaluation']['reasoning']}")

            # Add evaluation to state
            state['recommendation_evaluation'] = evaluation

        print(" All agents consulted (including setup_engineer) - completing workflow")
        return {
            "messages": [],
            "next_agent": "COMPLETE",
            "iteration": iteration
        }

    # If we have candidate recommendations, complete
    if state.get('candidate_recommendations'):
        print(f" Found {len(state['candidate_recommendations'])} recommendations - completing workflow")
        return {
            "messages": [],
            "next_agent": "COMPLETE",
            "iteration": iteration
        }

    llm = create_llm(temperature=0.1)  # Low temp for consistent routing

    # Build context message
    context_parts = []
    context_parts.append(f"DRIVER FEEDBACK: {state['driver_feedback']}")
    context_parts.append(f"\nITERATION: {iteration}/{max_iter}")
    context_parts.append(f"AGENTS CONSULTED: {', '.join(agents_consulted) if agents_consulted else 'none'}")

    # Add data status
    if state.get('telemetry_data') is not None:
        context_parts.append("\nOK: Telemetry data loaded")
    if state.get('statistical_analysis'):
        context_parts.append("OK: Statistical analysis complete")
    if state.get('knowledge_insights'):
        context_parts.append("OK: Knowledge insights gathered")
    if state.get('candidate_recommendations'):
        context_parts.append(f"OK: {len(state['candidate_recommendations'])} recommendations generated")

    context_parts.append("\nDECIDE: Which agent should work next, or is analysis COMPLETE?")

    messages = [
        SystemMessage(content=get_supervisor_prompt()),
        HumanMessage(content="\n".join(context_parts))
    ]

    # Get supervisor decision
    response = llm.invoke(messages)
    decision_text = response.content

    print(f"\n Supervisor Decision:")
    print(decision_text)

    # Parse the decision
    next_agent = _parse_supervisor_decision(decision_text)

    # Check iteration limit
    if iteration > max_iter:
        print(f"\n Max iterations ({max_iter}) reached - forcing completion")
        next_agent = "COMPLETE"

    return {
        "messages": [response],
        "next_agent": next_agent.lower() if next_agent != "COMPLETE" else "COMPLETE",
        "iteration": iteration
    }


def _parse_supervisor_decision(decision_text: str) -> str:
    """Parse supervisor's decision from response text"""
    # Look for NEXT_AGENT: pattern
    patterns = [
        r'NEXT_AGENT:\s*(\w+)',
        r'Route to:\s*(\w+)',
        r'Next:\s*(\w+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, decision_text, re.IGNORECASE)
        if match:
            agent = match.group(1).lower()
            if agent in ['data_analyst', 'knowledge_expert', 'setup_engineer', 'complete']:
                return agent

    # Default: if no clear routing, check keywords
    if 'complete' in decision_text.lower() or 'done' in decision_text.lower():
        return "COMPLETE"
    elif 'data' in decision_text.lower() or 'analyst' in decision_text.lower():
        return "data_analyst"
    elif 'knowledge' in decision_text.lower() or 'expert' in decision_text.lower():
        return "knowledge_expert"
    elif 'engineer' in decision_text.lower() or 'recommend' in decision_text.lower():
        return "setup_engineer"

    return "COMPLETE"  # Safety fallback


# ===== DATA ANALYST AGENT =====

@track_agent_metrics('data_analyst')
def data_analyst_node(state: RaceEngineerState) -> Dict[str, Any]:
    """
    Data Analyst agent loads and analyzes telemetry data.

    Uses tools to load data, assess quality, and run statistical analysis.
    """
    iteration = state['iteration'] + 1
    max_iter = state['max_iterations']
    print(f"\n[{iteration}/{max_iter}]  DATA ANALYST: Analyzing data...")

    llm = create_llm(temperature=0.3)

    # Bind tools
    tools = [load_telemetry, inspect_quality, clean_data, select_features,
             correlation_analysis, regression_analysis]
    llm_with_tools = llm.bind_tools(tools)

    # Build task description
    task_parts = []
    task_parts.append(f"Driver feedback: {state['driver_feedback']}")
    task_parts.append(f"Telemetry files: {len(state['telemetry_file_paths'])} files")

    # Check caching flags to prevent redundant analysis
    telemetry_data = state.get('telemetry_data')
    data_loaded = state.get('data_loaded', False)
    quality_assessed = state.get('quality_assessed', False)
    stats_complete = state.get('statistical_analysis_complete', False)

    if not data_loaded:
        # First time: Load everything
        task_parts.append("\nTASK: Load and analyze the telemetry data.")
        task_parts.append("1. Load the data")
        task_parts.append("2. Inspect quality")
        task_parts.append("3. Clean if needed")
        task_parts.append("4. Select relevant features")
        task_parts.append("5. Run correlation or regression analysis")
    elif not quality_assessed:
        # Data loaded but quality not checked
        task_parts.append(f"\nData already loaded: {telemetry_data.get('num_sessions', 0)} sessions")
        task_parts.append("\nTASK: Assess data quality.")
        task_parts.append("1. Call inspect_quality() with no parameters (data is auto-injected)")
    elif not stats_complete:
        # Quality checked, now analyze
        task_parts.append(f"\nData loaded and quality assessed: {telemetry_data.get('num_sessions', 0)} sessions")
        task_parts.append(f"Available parameters: {', '.join(telemetry_data.get('parameters', []))}")
        task_parts.append("\nTASK: Run statistical analysis.")
        task_parts.append("1. Call correlation_analysis with these parameters:")
        task_parts.append(f"   features={telemetry_data.get('parameters', [])}")
        task_parts.append("   target='fastest_time'")
        task_parts.append("   (data_dict will be auto-injected, don't pass it)")
        task_parts.append("\nIMPORTANT: Do NOT pass data_dict parameter - it is automatically provided.")
    else:
        # All analysis complete - skip redundant work
        print("  ✓ Analysis already complete - skipping redundant work (CACHE HIT)")
        return {
            "messages": [],
            "agents_consulted": state['agents_consulted'] + ['data_analyst'],
        }

    messages = [
        SystemMessage(content=get_data_analyst_prompt()),
        HumanMessage(content="\n".join(task_parts))
    ]

    # Agent decides which tools to use
    response = llm_with_tools.invoke(messages)

    # Execute tool calls if present
    new_messages = [response]
    updates = {"agents_consulted": state['agents_consulted'] + ['data_analyst']}

    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\n Calling {len(response.tool_calls)} tool(s)...")

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            print(f"   -> {tool_name}({list(tool_args.keys())})")

            # Execute the tool (inject state data if needed)
            tool_result = _execute_tool(tool_name, tool_args, tools, state)

            # Update state based on tool results
            if tool_name == 'load_telemetry' and 'data' in tool_result:
                updates['telemetry_data'] = tool_result
                updates['data_loaded'] = True  # Set flag to prevent reloading
                updates['tools_called'] = state['tools_called'] + ['load_telemetry']

            elif tool_name == 'inspect_quality':
                updates['data_quality_report'] = tool_result
                updates['quality_assessed'] = True  # Set flag to prevent re-assessment
                updates['tools_called'] = state['tools_called'] + ['inspect_quality']

            elif tool_name == 'select_features':
                updates['feature_analysis'] = tool_result
                updates['tools_called'] = state['tools_called'] + ['select_features']

            elif tool_name in ['correlation_analysis', 'regression_analysis']:
                updates['statistical_analysis'] = tool_result
                updates['statistical_analysis_complete'] = True  # Set flag to prevent re-analysis
                updates['tools_called'] = state['tools_called'] + [tool_name]

            # Add tool result to messages
            from langchain_core.messages import ToolMessage
            new_messages.append(ToolMessage(
                content=json.dumps(tool_result, default=str),
                tool_call_id=tool_call['id']
            ))

        # Get agent's summary after tools
        summary_response = llm.invoke(messages + new_messages)
        new_messages.append(summary_response)
        print(f"\n Summary: {summary_response.content}")

    updates['messages'] = new_messages

    return updates


# ===== KNOWLEDGE EXPERT AGENT =====

@track_agent_metrics('knowledge_expert')
def knowledge_expert_node(state: RaceEngineerState) -> Dict[str, Any]:
    """
    Knowledge Expert agent queries setup manuals and historical data.
    """
    iteration = state['iteration'] + 1
    max_iter = state['max_iterations']
    print(f"\n[{iteration}/{max_iter}]  KNOWLEDGE EXPERT: Consulting NASCAR manual...")

    llm = create_llm(temperature=0.3)

    # Bind tools
    tools = [query_setup_manual, search_history]
    llm_with_tools = llm.bind_tools(tools)

    # Extract complaint type from driver feedback
    complaint_type = _extract_complaint_type(state['driver_feedback'])

    task_parts = []
    task_parts.append(f"Driver complaint: {state['driver_feedback']}")
    task_parts.append(f"Identified issue: {complaint_type}")
    task_parts.append("\nTASK: Provide setup knowledge and historical context.")
    task_parts.append("Use query_setup_manual to get relevant guidance.")

    messages = [
        SystemMessage(content=get_knowledge_expert_prompt()),
        HumanMessage(content="\n".join(task_parts))
    ]

    response = llm_with_tools.invoke(messages)

    new_messages = [response]
    updates = {"agents_consulted": state['agents_consulted'] + ['knowledge_expert']}

    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\n Calling {len(response.tool_calls)} tool(s)...")

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            print(f"   -> {tool_name}({list(tool_args.keys())})")

            tool_result = _execute_tool(tool_name, tool_args, tools, state)

            if tool_name == 'query_setup_manual':
                updates['knowledge_insights'] = tool_result
                updates['tools_called'] = state['tools_called'] + ['query_setup_manual']

            from langchain_core.messages import ToolMessage
            new_messages.append(ToolMessage(
                content=json.dumps(tool_result, default=str),
                tool_call_id=tool_call['id']
            ))

        # Get summary
        summary_response = llm.invoke(messages + new_messages)
        new_messages.append(summary_response)
        print(f"\n Summary: {summary_response.content}")

    updates['messages'] = new_messages

    return updates


# ===== SETUP ENGINEER AGENT =====

@track_agent_metrics('setup_engineer')
def setup_engineer_node(state: RaceEngineerState) -> Dict[str, Any]:
    """Setup Engineer with explicit Think-Act-Observe loop"""

    print("\n" + "="*70)
    print("SETUP ENGINEER: Starting Think-Act-Observe Loop")
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
        print("THINK: Planning approach...")
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

        print(f"Strategy: {thinking.content}")

        # === ACT PHASE ===
        print("\nACT: Executing tools...")
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
                result = _execute_tool(tc['name'], tc['args'], [check_constraints, validate_physics], state)
                tool_results.append({tc['name']: result})
                print(f"  {tc['name']}: {str(result)}")

        # === OBSERVE PHASE ===
        print("\nOBSERVE: Reflecting on results...")
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

        print(f"Observation: {observation.content}")

        # Parse observation
        try:
            obs_data = json.loads(observation.content.strip().replace("```json", "").replace("```", ""))
            confidence = obs_data.get('confidence', 0.5)
            ready = obs_data.get('ready', False)

            print(f"  Confidence: {confidence:.1%}")

            if ready and confidence > 0.7:
                print("  Ready to recommend")
                break
            elif cycle == max_cycles:
                print("  Max cycles reached, proceeding with current confidence")
                break
            else:
                print(f"  Not ready (confidence: {confidence:.1%}), continuing...")
                context += f"\n\nCycle {cycle} findings: {obs_data.get('gaps', 'Refining approach')}"

        except:
            print("  Could not parse observation, proceeding")
            break

    # Generate final recommendation
    print("\nGenerating final recommendation...")
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
    """Generate structured final recommendation using all available insights"""

    # Try statistical analysis first
    stats = state.get('statistical_analysis') or {}
    all_impacts = stats.get('correlations') or stats.get('coefficients', {})

    if all_impacts:
        # Use statistical correlations
        sorted_params = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        top_param, top_impact = sorted_params[0]
        direction = "decrease" if top_impact > 0 else "increase"
        magnitude, unit = _estimate_magnitude(top_param)
        rationale = f"Statistical analysis shows {top_impact:+.3f} correlation with lap time"
        confidence = 0.8 if abs(top_impact) > 0.3 else 0.6
    else:
        # Fall back to knowledge expert insights
        knowledge = state.get('knowledge_insights', {})
        param_guidance = knowledge.get('parameter_guidance', {})

        if not param_guidance:
            # Last resort: provide general balance recommendations
            return {
                "primary": {
                    "parameter": "cross_weight",
                    "direction": "adjust",
                    "magnitude": "0.5",
                    "magnitude_unit": "%",
                    "confidence": 0.6,
                    "rationale": "For general handling issues, adjusting cross weight helps balance left/right grip. Start with small 0.5% adjustments and get driver feedback.",
                    "tool_validations": tool_results
                },
                "recommendations": [
                    {
                        "parameter": "cross_weight",
                        "direction": "adjust",
                        "magnitude": "0.5",
                        "magnitude_unit": "%"
                    },
                    {
                        "parameter": "tire_pressures",
                        "direction": "review",
                        "magnitude": "Check all corners",
                        "magnitude_unit": ""
                    },
                    {
                        "parameter": "spring_rates",
                        "direction": "review",
                        "magnitude": "Verify balance",
                        "magnitude_unit": ""
                    }
                ],
                "summary": "General handling: Start with cross weight adjustment and review basic balance"
            }

        # Use first parameter from NASCAR manual guidance
        top_param = list(param_guidance.keys())[0]
        guidance = param_guidance[top_param]
        direction = guidance.get('action', 'adjust')
        magnitude = guidance.get('magnitude', '1-2 units')

        # Extract unit from magnitude if present
        if 'PSI' in magnitude:
            unit = 'PSI'
            # Extract numeric value from magnitude string like "1.0-2.0 PSI"
            magnitude = magnitude.replace(' PSI', '').split('-')[0]
        elif 'lb/in' in magnitude:
            unit = 'lb/in'
            magnitude = magnitude.replace(' lb/in', '').split('-')[0]
        elif 'inches' in magnitude:
            unit = 'inches'
            magnitude = magnitude.replace(' inches', '').split('-')[0]
        else:
            unit = 'units'

        rationale = guidance.get('rationale', 'Based on NASCAR manual guidance')
        confidence = 0.75  # Medium-high confidence for manual guidance

    rec = {
        "primary": {
            "parameter": top_param,
            "direction": direction,
            "magnitude": magnitude,
            "magnitude_unit": unit,
            "confidence": confidence,
            "rationale": rationale,
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

    # Add secondary recommendations from knowledge insights
    if not all_impacts:
        knowledge = state.get('knowledge_insights', {})
        param_guidance = knowledge.get('parameter_guidance', {})
        for i, (param, guidance) in enumerate(list(param_guidance.items())[1:3]):  # Get 2nd and 3rd params
            rec["recommendations"].append({
                "parameter": param,
                "direction": guidance.get('action', 'adjust'),
                "magnitude": guidance.get('magnitude', 'TBD'),
                "rationale": guidance.get('rationale', '')
            })

    return rec


def _parse_recommendations_from_text(text: str, state: RaceEngineerState) -> List[Dict[str, Any]]:
    """
    Parse recommendations from LLM response text.
    Looks for patterns like: "decrease tire_psi_rr by 1.5 PSI"
    """
    import re

    recommendations = []

    # Pattern 1: "decrease/increase PARAM by AMOUNT UNIT"
    pattern1 = r'(decrease|increase|reduce|raise|lower)\s+(\w+)\s+by\s+([\d.]+)\s*(\w+)?'
    matches = re.finditer(pattern1, text, re.IGNORECASE)

    for match in matches:
        direction = match.group(1).lower()
        if direction in ['reduce', 'lower']:
            direction = 'decrease'
        elif direction in ['raise']:
            direction = 'increase'

        parameter = match.group(2)
        magnitude = float(match.group(3))
        unit = match.group(4) if match.group(4) else "units"

        rec = {
            "parameter": parameter,
            "direction": direction,
            "magnitude": magnitude,
            "magnitude_unit": unit,
            "rationale": "Based on statistical analysis and driver feedback",
            "confidence": 0.8,
            "expected_impact": "Improve lap time"
        }

        recommendations.append(rec)

    # FALLBACK: If no patterns found, use statistical analysis
    if not recommendations and state.get('statistical_analysis'):
        stats = state['statistical_analysis']
        all_impacts = stats.get('correlations') or stats.get('coefficients', {})

        if all_impacts:
            # Get top 3 parameters
            sorted_params = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

            for param, impact in sorted_params:
                direction = "decrease" if impact > 0 else "increase"
                magnitude, unit = _estimate_magnitude(param)

                rec = {
                    "parameter": param,
                    "direction": direction,
                    "magnitude": magnitude,
                    "magnitude_unit": unit,
                    "rationale": f"Correlation: {impact:+.3f}",
                    "confidence": 0.8 if abs(impact) > 0.3 else 0.6,
                    "expected_impact": f"Based on {impact:+.3f} correlation"
                }

                recommendations.append(rec)

    return recommendations


def _estimate_magnitude(parameter: str) -> tuple:
    """Estimate appropriate magnitude for a parameter"""
    if 'psi' in parameter.lower() or 'tire' in parameter.lower():
        return 1.5, "PSI"
    elif 'spring' in parameter.lower():
        return 25.0, "lb/in"
    elif 'cross_weight' in parameter.lower() or 'weight' in parameter.lower():
        return 0.5, "%"
    elif 'track_bar' in parameter.lower() or 'height' in parameter.lower():
        return 0.25, "inches"
    else:
        return 1.0, "units"


# ===== HELPER FUNCTIONS =====

def _execute_tool(tool_name: str, tool_args: Dict[str, Any], tools: List, state: Dict = None) -> Any:
    """Execute a tool by name with given arguments, auto-injecting state data if needed"""

    # Auto-inject telemetry_data for tools that need it
    data_tools = ['inspect_quality', 'clean_data', 'select_features', 'correlation_analysis', 'regression_analysis']
    if state and tool_name in data_tools and 'data_dict' not in tool_args:
        telemetry_data = state.get('telemetry_data')
        if telemetry_data:
            # Inject the telemetry data automatically
            tool_args = {**tool_args, 'data_dict': telemetry_data}

    for tool in tools:
        if tool.name == tool_name:
            try:
                result = tool.invoke(tool_args)
                return result
            except Exception as e:
                return {"error": str(e)}

    return {"error": f"Tool {tool_name} not found"}


def _extract_complaint_type(feedback: str) -> str:
    """Extract complaint type from driver feedback"""
    feedback_lower = feedback.lower()

    if any(word in feedback_lower for word in ['loose', 'oversteer', 'rear', 'slide']):
        return "oversteer"
    elif any(word in feedback_lower for word in ['tight', 'understeer', 'push', 'plow']):
        return "understeer"
    elif 'bottom' in feedback_lower:
        return "bottoming"
    else:
        return "general"


def _detect_conflicts(insights: Dict) -> List[str]:
    """Detect conflicts between agent insights"""
    conflicts = []

    # Example: Data says X but knowledge says Y
    data_insight = insights.get('data')
    data_top = data_insight.get('top_parameter') if isinstance(data_insight, dict) else None

    knowledge_insight = insights.get('knowledge')
    if isinstance(knowledge_insight, dict):
        knowledge_params = knowledge_insight.get('parameter_guidance', {})
        if data_top and data_top not in knowledge_params:
            conflicts.append(f"Data suggests {data_top} but knowledge has no guidance on it")

    # Check if recommendations conflict with constraints
    recs = insights.get('recommendations', [])
    if recs is None:
        recs = []
    for rec in recs:
        if isinstance(rec, dict) and rec.get('constraint_violations'):
            conflicts.append(f"{rec['parameter']}: Violates constraints")

    return conflicts


