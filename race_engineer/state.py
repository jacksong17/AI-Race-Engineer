"""
State schema for the Race Engineer agentic workflow.

Defines the complete state structure that flows through all agents.
"""

from typing import TypedDict, Annotated, List, Dict, Optional, Literal, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import pandas as pd


class RaceEngineerState(TypedDict):
    """
    Complete state for agentic race engineer workflow.

    This state is passed between all agents and maintains the complete
    context of the analysis session.
    """

    # ===== CONVERSATION MANAGEMENT =====
    messages: Annotated[List[BaseMessage], add_messages]
    """Full conversation history between agents and tools for coordination"""

    # ===== INPUT DATA =====
    driver_feedback: str
    """Original driver feedback text"""

    driver_feedback_parsed: Optional[Dict[str, Any]]
    """Structured driver feedback after parsing:
       - complaint_type: str (oversteer, understeer, bottoming, etc)
       - severity: str (minor, moderate, severe)
       - phase: str (entry, mid, exit, straight)
       - description: str
       - concerns: List[str]
    """

    telemetry_file_paths: List[str]
    """Paths to telemetry files (.ibt, .ldx, .csv)"""

    driver_constraints: Optional[Dict[str, Any]]
    """Driver-specified constraints:
       - parameters_at_limit: Dict[str, str]  # param -> 'min'|'max'
       - cannot_adjust: List[str]  # Parameters that cannot be changed
       - already_tried: List[str]  # Parameters already tested
       - priority_areas: List[str]  # Areas driver wants to focus on
    """

    session_config: Dict[str, Any]
    """Analysis configuration:
       - track: str (default: bristol)
       - car_class: str (default: nascar_truck)
       - conditions: str (dry, wet, temperature, etc)
       - analysis_mode: str (quick, standard, comprehensive)
    """

    # ===== LOADED DATA =====
    telemetry_data: Optional[Any]  # pd.DataFrame but using Any for typing
    """Loaded and cleaned telemetry data as DataFrame"""

    data_quality_report: Optional[Dict[str, Any]]
    """Data quality assessment:
       - num_sessions: int
       - lap_time_range: tuple (min, max)
       - outliers_detected: List[Dict]
       - missing_data: Dict[str, float]
       - usable_parameters: List[str]
       - quality_score: float (0-1)
    """

    # ===== ANALYSIS COMPLETION FLAGS (Prevent Redundant Work) =====
    data_loaded: bool
    """Flag: Has telemetry data been loaded?"""

    quality_assessed: bool
    """Flag: Has data quality assessment been completed?"""

    statistical_analysis_complete: bool
    """Flag: Has statistical analysis been performed?"""

    # ===== ANALYSIS RESULTS =====
    feature_analysis: Optional[Dict[str, Any]]
    """Feature selection and importance:
       - selected_features: List[str]
       - variance_scores: Dict[str, float]
       - relevance_to_complaint: Dict[str, float]
       - rejection_reasons: Dict[str, str]
    """

    statistical_analysis: Optional[Dict[str, Any]]
    """Statistical analysis results:
       - method: str (correlation, regression, mixed)
       - parameter_impacts: Dict[str, float]
       - confidence_scores: Dict[str, float]
       - p_values: Optional[Dict[str, float]]
       - r_squared: Optional[float]
       - significant_params: List[str]
    """

    knowledge_insights: Optional[Dict[str, Any]]
    """Setup manual and historical insights:
       - relevant_sections: List[str]
       - parameter_limits: Dict[str, Dict]
       - setup_principles: List[str]
       - similar_historical_cases: List[Dict]
       - successful_past_solutions: List[Dict]
    """

    # ===== RECOMMENDATIONS =====
    candidate_recommendations: List[Dict[str, Any]]
    """All proposed recommendations from agents:
       Each recommendation contains:
         - parameter: str
         - direction: str (increase/decrease)
         - magnitude: float
         - magnitude_unit: str (psi, lb/in, %, inches)
         - rationale: str
         - confidence: float (0-1)
         - agent_source: str
         - expected_impact: Optional[float]
    """

    validated_recommendations: Optional[Dict[str, Any]]
    """Recommendations after validation:
       - primary: Dict (main recommendation)
       - secondary: List[Dict] (supporting recommendations)
       - validation_results: List[Dict]
       - warnings: List[str]
       - constraints_checked: List[str]
       - physics_validated: bool
    """

    final_recommendation: Optional[Dict[str, Any]]
    """Final synthesized recommendation for output:
       - primary_change: Dict
       - secondary_changes: List[Dict]
       - expected_impact: str
       - confidence: float
       - rationale: str
       - caveats: List[str]
    """

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

    # ===== WORKFLOW CONTROL =====
    next_agent: Optional[str]
    """Supervisor's decision on next agent to call:
       - data_analyst
       - knowledge_expert
       - setup_engineer
       - COMPLETE
    """

    iteration: int
    """Current iteration count (starts at 0)"""

    max_iterations: int
    """Maximum allowed iterations to prevent infinite loops (default: 5)"""

    agents_consulted: List[str]
    """Track which agents have been called this session"""

    tools_called: List[str]
    """Track which tools have been used"""

    workflow_status: Literal["active", "completed", "error"]
    """Current workflow state"""

    # ===== ERROR HANDLING =====
    errors: List[Dict[str, Any]]
    """Any errors encountered:
       - agent: str
       - error_type: str
       - message: str
       - recoverable: bool
       - timestamp: str
    """

    warnings: List[str]
    """Non-fatal warnings collected during analysis"""

    # ===== SESSION METADATA =====
    session_id: str
    """Unique session identifier (UUID)"""

    session_timestamp: str
    """ISO timestamp when session started"""

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


def create_initial_state(
    driver_feedback: str,
    telemetry_files: List[str],
    constraints: Optional[Dict] = None,
    config: Optional[Dict] = None,
    session_id: Optional[str] = None
) -> RaceEngineerState:
    """
    Create initial state for a new analysis session.

    Args:
        driver_feedback: Raw driver feedback text
        telemetry_files: List of file paths to telemetry data
        constraints: Optional driver constraints
        config: Optional session configuration
        session_id: Optional session ID (generated if not provided)

    Returns:
        Initialized RaceEngineerState
    """
    from datetime import datetime
    import uuid

    if session_id is None:
        session_id = str(uuid.uuid4())

    if config is None:
        config = {
            "track": "bristol",
            "car_class": "nascar_truck",
            "conditions": "dry",
            "analysis_mode": "standard"
        }

    return {
        # Conversation
        "messages": [],

        # Input
        "driver_feedback": driver_feedback,
        "driver_feedback_parsed": None,
        "telemetry_file_paths": telemetry_files,
        "driver_constraints": constraints,
        "session_config": config,

        # Loaded data
        "telemetry_data": None,
        "data_quality_report": None,

        # Analysis completion flags
        "data_loaded": False,
        "quality_assessed": False,
        "statistical_analysis_complete": False,

        # Analysis
        "feature_analysis": None,
        "statistical_analysis": None,
        "knowledge_insights": None,

        # Recommendations
        "candidate_recommendations": [],
        "validated_recommendations": None,
        "final_recommendation": None,
        "previous_recommendations": [],
        "parameter_adjustment_history": {},
        "recommendation_stats": {
            "total_proposed": 0,
            "unique_accepted": 0,
            "duplicates_filtered": 0,
            "constraint_violations_caught": 0,
            "parameters_touched": []
        },

        # Workflow control
        "next_agent": None,
        "iteration": 0,
        "max_iterations": 5,
        "agents_consulted": [],
        "tools_called": [],
        "workflow_status": "active",

        # Error handling
        "errors": [],
        "warnings": [],

        # Metadata
        "session_id": session_id,
        "session_timestamp": datetime.utcnow().isoformat(),

        # Metrics
        "agent_metrics": {},
        "total_cost_estimate": 0.0
    }
