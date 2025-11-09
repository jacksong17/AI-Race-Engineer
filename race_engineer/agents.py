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
    print(f"\n[{iteration}/{max_iter}] 🎯 SUPERVISOR: Routing decision...")

    # CRITICAL FIX: Check if we already have recommendations and should complete
    agents_consulted = state.get('agents_consulted', [])

    # If setup_engineer already ran, we should complete
    if 'setup_engineer' in agents_consulted:
        print("✅ All agents consulted (including setup_engineer) - completing workflow")
        return {
            "messages": [],
            "next_agent": "COMPLETE",
            "iteration": iteration
        }

    # If we have candidate recommendations, complete
    if state.get('candidate_recommendations'):
        print(f"✅ Found {len(state['candidate_recommendations'])} recommendations - completing workflow")
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
    if iteration > max_iter:
        print(f"\n⚠️  Max iterations ({max_iter}) reached - forcing completion")
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
    print(f"\n[{iteration}/{max_iter}] 📊 DATA ANALYST: Analyzing data...")

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
    iteration = state['iteration'] + 1
    max_iter = state['max_iterations']
    print(f"\n[{iteration}/{max_iter}] 📚 KNOWLEDGE EXPERT: Consulting NASCAR manual...")

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

    NOW WITH DEDUPLICATION - Never suggests the same thing twice!
    """
    iteration = state['iteration'] + 1
    max_iter = state['max_iterations']
    print(f"\n[{iteration}/{max_iter}] 🔧 SETUP ENGINEER: Generating recommendations...")

    # CRITICAL FIX: If we already have candidate recommendations, don't regenerate
    existing_candidates = state.get('candidate_recommendations', [])
    if existing_candidates:
        print(f"\n📋 Found {len(existing_candidates)} existing recommendation(s) - using those")
        # Just ensure final_recommendation is set from existing candidates
        if not state.get('final_recommendation'):
            return {
                'agents_consulted': state['agents_consulted'] + ['setup_engineer'],
                'final_recommendation': {
                    "primary": existing_candidates[0] if existing_candidates else None,
                    "secondary": existing_candidates[1:] if len(existing_candidates) > 1 else [],
                    "summary": f"Using {len(existing_candidates)} existing recommendation(s)",
                    "num_unique": len(existing_candidates),
                    "num_filtered": 0
                }
            }
        else:
            # Already have both candidates and final rec, just mark engineer as consulted
            return {
                'agents_consulted': state['agents_consulted'] + ['setup_engineer']
            }

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

    # CRITICAL FIX: Ensure final_recommendation is ALWAYS set
    if 'final_recommendation' not in updates:
        # Try to construct from statistical analysis as fallback
        analysis = state.get('statistical_analysis', {})
        if analysis.get('top_parameter'):
            param = analysis['top_parameter']
            corr = analysis.get('top_correlation', 0)
            updates['final_recommendation'] = {
                "primary": {
                    'parameter': param,
                    'direction': 'decrease' if corr > 0 else 'increase',
                    'magnitude': 1.5 if 'psi' in param.lower() else 25,
                    'magnitude_unit': 'PSI' if 'psi' in param.lower() else 'lb/in',
                    'rationale': f'Based on correlation analysis ({corr:.3f})',
                    'confidence': 0.6
                },
                "summary": f"Recommended: {'Decrease' if corr > 0 else 'Increase'} {param} based on telemetry correlation",
                "num_unique": 1,
                "num_filtered": 0
            }
            print("\n⚠️  No agent recommendations - using statistical fallback")
        else:
            # Last resort: indicate no recommendation possible
            updates['final_recommendation'] = {
                "primary": None,
                "summary": "Unable to generate recommendation with available data",
                "num_unique": 0,
                "num_filtered": 0
            }
            print("\n⚠️  Unable to generate recommendations - insufficient data")

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
