"""
LangGraph workflow construction for the Race Engineer system.

Defines the graph structure, routing logic, and execution flow.
"""

import sys
import io
from langgraph.graph import StateGraph, END

# Fix Windows Unicode issues
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
try:
    from langgraph_checkpoint_sqlite import SqliteSaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as SqliteSaver
from pathlib import Path
from race_engineer.state import RaceEngineerState
from race_engineer.agents import (
    supervisor_node,
    data_analyst_node,
    knowledge_expert_node,
    setup_engineer_node
)


def create_race_engineer_graph(checkpointer=None):
    """
    Build the complete Race Engineer agentic workflow.

    Graph structure:
        START → supervisor
            ↓
        supervisor → [data_analyst | knowledge_expert | setup_engineer | END]
            ↓
        specialist agents → supervisor (loop back)
            ↓
        supervisor → END (when complete)

    Args:
        checkpointer: Optional checkpointer for persistence

    Returns:
        Compiled LangGraph application
    """

    # Initialize the state graph
    workflow = StateGraph(RaceEngineerState)

    # ===== ADD NODES =====

    # Supervisor orchestrates everything
    workflow.add_node("supervisor", supervisor_node)

    # Specialist agents
    workflow.add_node("data_analyst", data_analyst_node)
    workflow.add_node("knowledge_expert", knowledge_expert_node)
    workflow.add_node("setup_engineer", setup_engineer_node)

    # ===== DEFINE EDGES =====

    # Entry point is always supervisor
    workflow.set_entry_point("supervisor")

    # Supervisor routes to specialists or completes
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "data_analyst": "data_analyst",
            "knowledge_expert": "knowledge_expert",
            "setup_engineer": "setup_engineer",
            "complete": END
        }
    )

    # All specialists route back to supervisor for next decision
    workflow.add_edge("data_analyst", "supervisor")
    workflow.add_edge("knowledge_expert", "supervisor")
    workflow.add_edge("setup_engineer", "supervisor")

    # ===== COMPILE =====

    # Use checkpointer if provided, otherwise no persistence
    if checkpointer is None:
        app = workflow.compile()
    else:
        app = workflow.compile(checkpointer=checkpointer)

    return app


def route_supervisor(state: RaceEngineerState) -> str:
    """
    Routing logic from supervisor to specialists or completion.

    Args:
        state: Current state

    Returns:
        Next node to execute ("data_analyst", "knowledge_expert", "setup_engineer", or "complete")
    """

    # Check iteration limit first
    if state['iteration'] >= state['max_iterations']:
        print("⚠️  Maximum iterations reached - completing analysis")
        return "complete"

    # Get supervisor's decision
    next_agent = state.get('next_agent', '').lower()

    # Route based on decision
    if next_agent == 'complete':
        return "complete"
    elif next_agent == 'data_analyst':
        return "data_analyst"
    elif next_agent == 'knowledge_expert':
        return "knowledge_expert"
    elif next_agent == 'setup_engineer':
        return "setup_engineer"
    else:
        # Safety fallback: if unclear, complete
        print(f"⚠️  Unclear routing decision: '{next_agent}' - completing")
        return "complete"


def create_checkpointer(db_path: str = None) -> SqliteSaver:
    """
    Create SQLite checkpointer for persistence.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SqliteSaver instance
    """
    if db_path is None:
        # Default location
        db_path = Path(__file__).parent.parent / "data" / "knowledge" / "checkpoints.db"

    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    return SqliteSaver.from_conn_string(str(db_path))


def get_graph_visualization(app) -> str:
    """
    Get a visualization of the graph structure.

    Args:
        app: Compiled LangGraph application

    Returns:
        Mermaid diagram as string
    """
    try:
        return app.get_graph().draw_mermaid()
    except Exception as e:
        return f"Visualization error: {str(e)}"


def print_graph_info(app):
    """
    Print information about the compiled graph.

    Args:
        app: Compiled LangGraph application
    """
    print("\n" + "="*70)
    print("GRAPH STRUCTURE")
    print("="*70)

    try:
        graph = app.get_graph()

        print("\nNodes:")
        for node_id in graph.nodes:
            print(f"  • {node_id}")

        print("\nEdges:")
        for edge in graph.edges:
            print(f"  • {edge}")

        print("\nWorkflow:")
        print("  1. START → supervisor")
        print("  2. supervisor → [data_analyst | knowledge_expert | setup_engineer | END]")
        print("  3. specialists → supervisor (iterative loop)")
        print("  4. supervisor → END (when complete)")

    except Exception as e:
        print(f"Error displaying graph info: {e}")

    print("="*70 + "\n")


# ===== EXECUTION UTILITIES =====

def run_workflow(
    app,
    driver_feedback: str,
    telemetry_files: list,
    constraints: dict = None,
    config: dict = None,
    verbose: bool = True
):
    """
    Execute the workflow with given inputs.

    Args:
        app: Compiled LangGraph application
        driver_feedback: Driver feedback text
        telemetry_files: List of telemetry file paths
        constraints: Optional driver constraints
        config: Optional session configuration
        verbose: Print progress updates

    Returns:
        Final state after workflow completion
    """
    from race_engineer.state import create_initial_state

    # Create initial state
    initial_state = create_initial_state(
        driver_feedback=driver_feedback,
        telemetry_files=telemetry_files,
        constraints=constraints,
        config=config
    )

    if verbose:
        print("\n" + "🏁"*35)
        print("AI RACE ENGINEER - Starting Analysis")
        print("🏁"*35)
        print(f"\nDriver Feedback: {driver_feedback}")
        print(f"Telemetry Files: {len(telemetry_files)} files")
        print(f"Session ID: {initial_state['session_id']}")

    # Execute workflow
    try:
        # For workflows without checkpointing, use invoke
        final_state = app.invoke(initial_state)

        if verbose:
            print("\n" + "✅"*35)
            print("ANALYSIS COMPLETE")
            print("✅"*35)

            # Print summary
            if final_state.get('final_recommendation'):
                rec = final_state['final_recommendation']
                print("\n📋 FINAL RECOMMENDATION:")
                print(rec.get('summary', 'See detailed output'))

            if final_state.get('errors'):
                print("\n⚠️  ERRORS:")
                for error in final_state['errors']:
                    print(f"  • {error}")

        return final_state

    except Exception as e:
        print(f"\n❌ Workflow execution failed: {str(e)}")
        raise


def run_workflow_streaming(
    app,
    driver_feedback: str,
    telemetry_files: list,
    constraints: dict = None,
    config: dict = None
):
    """
    Execute workflow with streaming output.

    Args:
        app: Compiled LangGraph application
        driver_feedback: Driver feedback text
        telemetry_files: List of telemetry file paths
        constraints: Optional driver constraints
        config: Optional session configuration

    Yields:
        State updates as they occur
    """
    from race_engineer.state import create_initial_state

    initial_state = create_initial_state(
        driver_feedback=driver_feedback,
        telemetry_files=telemetry_files,
        constraints=constraints,
        config=config
    )

    print("\n" + "🏁"*35)
    print("AI RACE ENGINEER - Streaming Analysis")
    print("🏁"*35)

    try:
        for event in app.stream(initial_state):
            # event is a dict with node name as key
            for node_name, node_output in event.items():
                print(f"\n[{node_name.upper()}] Node executed")
                yield node_output

    except Exception as e:
        print(f"\n❌ Streaming failed: {str(e)}")
        raise
