"""
AI Race Engineer - Production Agentic System

A LangGraph-based multi-agent system for NASCAR racing telemetry analysis.
"""

__version__ = "2.0.0"

from race_engineer.graph import create_race_engineer_graph

# Create and export the default app instance
app = create_race_engineer_graph()
