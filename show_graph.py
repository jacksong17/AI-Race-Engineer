"""
Display the LangGraph workflow structure
Perfect for showing architecture to technical audience
"""

from race_engineer import create_race_engineer_workflow

print("="*70)
print("  BRISTOL AI RACE ENGINEER - WORKFLOW GRAPH")
print("="*70)
print()

# Create the workflow
app = create_race_engineer_workflow()

# Show ASCII representation
print("ASCII Representation:")
print("-"*70)
try:
    print(app.get_graph().draw_ascii())
except:
    print("Note: ASCII graph not available in this LangGraph version")
print()

# Show graph structure details
print("Graph Structure:")
print("-"*70)
graph = app.get_graph()

print(f"\nNodes ({len(graph.nodes)}):")
for node_name in graph.nodes:
    print(f"  - {node_name}")

print(f"\nEdges ({len(graph.edges)}):")
for edge in graph.edges:
    source = edge.source if hasattr(edge, 'source') else 'start'
    target = edge.target if hasattr(edge, 'target') else 'end'
    print(f"  - {source} -> {target}")

print()
print("Workflow Characteristics:")
print("-"*70)
print(f"  Entry Point: telemetry")
print(f"  Error Handling: Explicit error node with conditional routing")
print(f"  State Type: TypedDict (strongly typed)")
print(f"  Execution: Synchronous, deterministic")
print(f"  Extensibility: Add nodes/edges without modifying existing code")
print()

print("To visualize graphically (requires graphviz):")
print("  pip install pygraphviz")
print("  # Then use app.get_graph().draw_mermaid_png()")
print()
print("="*70)
