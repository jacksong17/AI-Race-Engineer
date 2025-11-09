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
    print("\n" + "="*70)
    print("🎯 SUPERVISOR: Orchestrating analysis workflow")
    print("="*70)

    llm = create_llm(temperature=0.1)  # Low temp for consistent routing

    # Build context message
    context_parts = []
    context_parts.append(f"DRIVER FEEDBACK: {state['driver_feedback']}")
    context_parts.append(f"\nITERATION: {state['iteration']}/{state['max_iterations']}")
    context_parts.append(f"AGENTS CONSULTED: {', '.join(state['agents_consulted']) if state['agents_consulted'] else 'none'}")

    # Add data status
    if state.get('telemetry_data'):
        context_parts.append("\n✓ Telemetry data loaded")
    if state.get('statistical_analysis'):
        context_parts.append("✓ Statistical analysis complete")
    if state.get('knowledge_insights'):
        context_parts.append("✓ Knowledge insights gathered")
    if state.get('candidate_recommendations'):
        context_parts.append(f"✓ {len(state['candidate_recommendations'])} recommendations generated")

    context_parts.append("\nDECIDE: Which agent should work next, or is analysis COMPLETE?")

    messages = [
        SystemMessage(content=get_supervisor_prompt()),
        HumanMessage(content="\n".join(context_parts))
    ]

    # Get supervisor decision
    response = llm.invoke(messages)
    decision_text = response.content

    print(f"\n📋 Supervisor Decision:")
    print(decision_text)

    # Parse the decision
    next_agent = _parse_supervisor_decision(decision_text)

    # Check iteration limit
    if state['iteration'] >= state['max_iterations']:
        print(f"\n⚠️  Max iterations ({state['max_iterations']}) reached - forcing completion")
        next_agent = "COMPLETE"

    return {
        "messages": [response],
        "next_agent": next_agent.lower() if next_agent != "COMPLETE" else "COMPLETE",
        "iteration": state['iteration'] + 1
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
    print("\n" + "="*70)
    print("📊 DATA ANALYST: Analyzing telemetry data")
    print("="*70)

    llm = create_llm(temperature=0.3)

    # Bind tools
    tools = [load_telemetry, inspect_quality, clean_data, select_features,
             correlation_analysis, regression_analysis]
    llm_with_tools = llm.bind_tools(tools)

    # Build task description
    task_parts = []
    task_parts.append(f"Driver feedback: {state['driver_feedback']}")
    task_parts.append(f"Telemetry files: {len(state['telemetry_file_paths'])} files")

    if not state.get('telemetry_data'):
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
        print(f"\n🔧 Calling {len(response.tool_calls)} tool(s)...")

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            print(f"   → {tool_name}({list(tool_args.keys())})")

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
        print(f"\n📝 Summary: {summary_response.content[:200]}...")

    updates['messages'] = new_messages

    return updates


# ===== KNOWLEDGE EXPERT AGENT =====

def knowledge_expert_node(state: RaceEngineerState) -> Dict[str, Any]:
    """
    Knowledge Expert agent queries setup manuals and historical data.
    """
    print("\n" + "="*70)
    print("📚 KNOWLEDGE EXPERT: Consulting setup knowledge")
    print("="*70)

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
        print(f"\n🔧 Calling {len(response.tool_calls)} tool(s)...")

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            print(f"   → {tool_name}({list(tool_args.keys())})")

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
        print(f"\n📝 Summary: {summary_response.content[:200]}...")

    updates['messages'] = new_messages

    return updates


# ===== SETUP ENGINEER AGENT =====

def setup_engineer_node(state: RaceEngineerState) -> Dict[str, Any]:
    """
    Setup Engineer agent generates specific recommendations.
    """
    print("\n" + "="*70)
    print("🔧 SETUP ENGINEER: Generating recommendations")
    print("="*70)

    llm = create_llm(temperature=0.3)

    # Bind tools
    tools = [check_constraints, validate_physics, visualize_impacts]
    llm_with_tools = llm.bind_tools(tools)

    # Build comprehensive context
    task_parts = []
    task_parts.append(f"Driver feedback: {state['driver_feedback']}")

    if state.get('statistical_analysis'):
        stats = state['statistical_analysis']
        task_parts.append(f"\nStatistical Analysis ({stats.get('method', 'unknown')}):")
        if 'top_parameter' in stats:
            task_parts.append(f"  Top parameter: {stats['top_parameter']}")
            if 'top_correlation' in stats:
                task_parts.append(f"  Correlation: {stats['top_correlation']:.3f}")

    if state.get('knowledge_insights'):
        task_parts.append("\nSetup knowledge consulted ✓")

    task_parts.append("\nTASK: Generate specific setup recommendations.")
    task_parts.append("1. Determine primary recommendation based on analysis")
    task_parts.append("2. Use check_constraints to validate")
    task_parts.append("3. Provide specific magnitude (PSI, lb/in, %, etc)")
    task_parts.append("4. Optionally create visualization")

    if state.get('driver_constraints'):
        task_parts.append(f"\nConstraints: {state['driver_constraints']}")

    messages = [
        SystemMessage(content=get_setup_engineer_prompt()),
        HumanMessage(content="\n".join(task_parts))
    ]

    response = llm_with_tools.invoke(messages)

    new_messages = [response]
    updates = {"agents_consulted": state['agents_consulted'] + ['setup_engineer']}

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

        # Get final recommendations
        summary_response = llm.invoke(messages + new_messages)
        new_messages.append(summary_response)
        print(f"\n📝 Recommendations: {summary_response.content[:300]}...")

        # Parse recommendations from response
        recommendations = _parse_recommendations(summary_response.content, state)
        if recommendations:
            updates['candidate_recommendations'] = recommendations
            updates['final_recommendation'] = {
                "primary": recommendations[0] if recommendations else None,
                "summary": summary_response.content
            }

    updates['messages'] = new_messages

    return updates


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


def _parse_recommendations(text: str, state: RaceEngineerState) -> List[Dict[str, Any]]:
    """Parse recommendations from agent response text"""
    recommendations = []

    # Try to extract structured recommendations
    # This is a simplified parser - could be enhanced with more robust parsing

    # Look for parameter names and actions
    if state.get('statistical_analysis'):
        stats = state['statistical_analysis']
        top_param = stats.get('top_parameter')

        if top_param:
            # Create primary recommendation
            direction = "increase" if stats.get('top_correlation', 0) < 0 else "decrease"

            rec = {
                "parameter": top_param,
                "direction": direction,
                "magnitude": 1.5 if 'psi' in top_param.lower() else 25,
                "magnitude_unit": "PSI" if 'psi' in top_param.lower() else "lb/in",
                "rationale": f"Statistical analysis shows {stats.get('method', 'correlation')} of {stats.get('top_correlation', 0):.3f}",
                "confidence": 0.8,
                "agent_source": "setup_engineer"
            }

            recommendations.append(rec)

    return recommendations
