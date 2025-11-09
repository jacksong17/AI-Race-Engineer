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
    visualize_impacts,
    save_session
)
from race_engineer.recommendation_deduplicator import check_duplicate, filter_unique
import re
import json


def create_llm(temperature: float = 0.3) -> ChatAnthropic:
    """
    Create Claude LLM instance.

    Using Haiku for cost efficiency while maintaining quality.
    """
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=temperature
    )


# ===== SUPERVISOR AGENT =====

def supervisor_node(state: RaceEngineerState) -> Dict[str, Any]:
    """
    Supervisor agent orchestrates the workflow.

    Decides which specialist agent to call next or when to complete.
    """
    iteration = state['iteration'] + 1
    max_iter = state['max_iterations']
    print(f"\n[{iteration}/{max_iter}]  SUPERVISOR: Routing decision...")

    # CRITICAL FIX: Check if we already have recommendations and should complete
    agents_consulted = state.get('agents_consulted', [])

    # If setup_engineer already ran, we should complete
    if 'setup_engineer' in agents_consulted:
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

    if state.get('telemetry_data') is None:
        task_parts.append("\nTASK: Load and analyze the telemetry data.")
        task_parts.append("1. Load the data")
        task_parts.append("2. Inspect quality")
        task_parts.append("3. Clean if needed")
        task_parts.append("4. Select relevant features")
        task_parts.append("5. Run correlation or regression analysis")
    else:
        task_parts.append("\nData already loaded. Perform additional analysis if needed.")

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

            # Execute the tool
            tool_result = _execute_tool(tool_name, tool_args, tools)

            # Update state based on tool results
            if tool_name == 'load_telemetry' and 'data' in tool_result:
                updates['telemetry_data'] = tool_result
                updates['tools_called'] = state['tools_called'] + ['load_telemetry']

            elif tool_name == 'inspect_quality':
                updates['data_quality_report'] = tool_result
                updates['tools_called'] = state['tools_called'] + ['inspect_quality']

            elif tool_name == 'select_features':
                updates['feature_analysis'] = tool_result
                updates['tools_called'] = state['tools_called'] + ['select_features']

            elif tool_name in ['correlation_analysis', 'regression_analysis']:
                updates['statistical_analysis'] = tool_result
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
        print(f"\n Summary: {summary_response.content[:200]}...")

    updates['messages'] = new_messages

    return updates


# ===== KNOWLEDGE EXPERT AGENT =====

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

            tool_result = _execute_tool(tool_name, tool_args, tools)

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
        print(f"\n Summary: {summary_response.content[:200]}...")

    updates['messages'] = new_messages

    return updates


# ===== SETUP ENGINEER AGENT =====

def setup_engineer_node(state: RaceEngineerState) -> Dict[str, Any]:
    """
    Setup Engineer agent generates specific recommendations.
    """
    print("\n" + "="*70)
    print("SETUP ENGINEER: Generating recommendations")
    print("="*70)

    llm = create_llm(temperature=0.3)

    # Check for previous recommendations
    previous_recs = state.get('previous_recommendations', [])
    already_recommended_params = {rec.get('parameter') for rec in previous_recs}

    print(f"\nPrevious recommendations: {len(previous_recs)}")
    if previous_recs:
        print("   Already recommended:")
        for rec in previous_recs:
            print(f"      - {rec.get('parameter')}: {rec.get('direction')} by {rec.get('magnitude')}")

    # Build task with EXPLICIT instruction format
    task_parts = []
    task_parts.append(f"Driver feedback: {state['driver_feedback']}")

    # Add statistical analysis if available
    if state.get('statistical_analysis'):
        stats = state['statistical_analysis']
        all_impacts = stats.get('correlations') or stats.get('coefficients', {})

        if all_impacts:
            sorted_impacts = sorted(all_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
            task_parts.append("\nStatistical Analysis Results:")
            for param, impact in sorted_impacts[:5]:
                direction = "Reduce" if impact > 0 else "Increase"
                task_parts.append(f"  {direction} {param}: {impact:+.3f}")

    # Add knowledge insights
    if state.get('knowledge_insights'):
        task_parts.append("\nNASCAR manual guidance consulted")

    # CRITICAL: Add explicit format instruction
    task_parts.append("\nGENERATE RECOMMENDATION:")
    task_parts.append("Based on the analysis above, recommend ONE specific setup change.")
    task_parts.append("Format: 'Recommend: [increase/decrease] [parameter] by [amount] [unit]'")
    task_parts.append("Example: 'Recommend: decrease tire_psi_rr by 1.5 PSI'")

    messages = [
        SystemMessage(content=get_setup_engineer_prompt()),
        HumanMessage(content="\n".join(task_parts))
    ]

    # Get recommendation WITHOUT tools first
    response = llm.invoke(messages)

    print(f"\nEngineer Response: {response.content[:300]}...")

    # Parse recommendation from response
    recommendations = _parse_recommendations_from_text(response.content, state)

    updates = {
        "agents_consulted": state['agents_consulted'] + ['setup_engineer'],
        "messages": [response]
    }

    # If we got recommendations, process them
    if recommendations:
        print(f"\nParsed {len(recommendations)} recommendation(s)")

        # Deduplicate if there are previous recommendations
        if previous_recs:
            dedup_result = filter_unique(recommendations, previous_recs)
            unique_recs = dedup_result['unique']
            duplicate_recs = dedup_result['duplicates']

            print(f"   Unique: {len(unique_recs)}, Duplicates: {len(duplicate_recs)}")
            recommendations = unique_recs

        # Update state with recommendations
        all_previous = previous_recs.copy()
        for rec in recommendations:
            rec['iteration_made'] = state['iteration']
            rec['agent_source'] = 'setup_engineer'
            all_previous.append(rec)

        updates['previous_recommendations'] = all_previous
        updates['candidate_recommendations'] = recommendations

        # Set final recommendation
        if recommendations:
            primary = recommendations[0]
            updates['final_recommendation'] = {
                "primary": primary,
                "secondary": recommendations[1:] if len(recommendations) > 1 else [],
                "summary": response.content,
                "num_unique": len(recommendations),
                "num_filtered": 0
            }

            print(f"\nPRIMARY: {primary['direction']} {primary['parameter']} by {primary['magnitude']} {primary['magnitude_unit']}")

    else:
        # FALLBACK: Create recommendation from statistical analysis
        print("\nNo recommendations parsed from LLM response - using fallback")

        if state.get('statistical_analysis'):
            stats = state['statistical_analysis']
            all_impacts = stats.get('correlations') or stats.get('coefficients', {})

            if all_impacts:
                # Get top parameter by absolute impact
                top_param, top_impact = max(all_impacts.items(), key=lambda x: abs(x[1]))

                # Determine direction
                direction = "decrease" if top_impact > 0 else "increase"

                # Estimate magnitude
                magnitude, unit = _estimate_magnitude(top_param)

                fallback_rec = {
                    "parameter": top_param,
                    "direction": direction,
                    "magnitude": magnitude,
                    "magnitude_unit": unit,
                    "rationale": f"Statistical analysis shows {top_impact:+.3f} correlation with lap time",
                    "confidence": 0.8 if abs(top_impact) > 0.3 else 0.6,
                    "expected_impact": f"Correlation: {top_impact:+.3f}"
                }

                updates['final_recommendation'] = {
                    "primary": fallback_rec,
                    "secondary": [],
                    "summary": f"Primary recommendation: {direction} {top_param} by {magnitude} {unit}",
                    "num_unique": 1,
                    "num_filtered": 0
                }

                updates['candidate_recommendations'] = [fallback_rec]

                print(f"FALLBACK: {direction} {top_param} by {magnitude} {unit}")

    return updates


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

def _execute_tool(tool_name: str, tool_args: Dict[str, Any], tools: List) -> Any:
    """Execute a tool by name with given arguments"""
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


