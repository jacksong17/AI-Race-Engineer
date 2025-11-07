"""
Session Memory Manager for AI Race Engineer

Manages persistent session storage across multiple testing stints.
Enables learning from previous sessions and tracking convergence toward optimal setup.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class SessionManager:
    """
    Manages session history and learning metrics for iterative testing.

    Features:
    - Persist sessions across program runs
    - Load historical context for informed decision-making
    - Calculate learning metrics (convergence, effectiveness, patterns)
    - Track recommendation outcomes
    """

    def __init__(self, storage_dir: Path = None):
        """
        Initialize SessionManager with storage directory.

        Args:
            storage_dir: Directory to store session files (default: ./sessions)
        """
        self.storage_dir = storage_dir or Path("sessions")
        self.storage_dir.mkdir(exist_ok=True)
        self.metrics_file = self.storage_dir / "learning_metrics.json"

    def save_session(self, state: Dict, session_id: str = None) -> str:
        """
        Save completed session to persistent storage.

        Args:
            state: RaceEngineerState dictionary containing analysis results
            session_id: Optional session identifier (auto-generated if not provided)

        Returns:
            session_id: The ID of the saved session
        """
        if session_id is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session_id = f"session_{timestamp}"

        # Build session summary (exclude large DataFrames)
        session_data = {
            "session_id": session_id,
            "timestamp": state.get('session_timestamp', datetime.now().isoformat()),
            "driver_feedback": state.get('driver_feedback', {}),
            "driver_diagnosis": state.get('driver_diagnosis', {}),
            "data_quality_decision": state.get('data_quality_decision', ''),
            "analysis_strategy": state.get('analysis_strategy', ''),
            "selected_features": state.get('selected_features', []),
            "analysis": state.get('analysis', {}),
            "recommendation": state.get('recommendation', ''),
            "outcome_feedback": state.get('outcome_feedback', None),
        }

        # Add data summary if raw_setup_data exists
        if state.get('raw_setup_data') is not None:
            df = state['raw_setup_data']
            session_data["data_summary"] = {
                "num_sessions_analyzed": len(df),
                "best_lap_time": float(df['fastest_time'].min()) if 'fastest_time' in df.columns else None,
                "improvement_range": float(df['fastest_time'].max() - df['fastest_time'].min()) if 'fastest_time' in df.columns else None,
            }

        # Save to file
        session_file = self.storage_dir / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        print(f"[SESSION MANAGER] Saved session: {session_id}")

        # Update learning metrics
        self._update_learning_metrics(session_data)

        return session_id

    def load_session_history(self, limit: int = 5) -> List[Dict]:
        """
        Load previous sessions for context.

        Args:
            limit: Maximum number of recent sessions to load (default: 5)

        Returns:
            List of session dictionaries, sorted by timestamp (most recent first)
        """
        session_files = sorted(
            self.storage_dir.glob("session_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        sessions = []
        for session_file in session_files[:limit]:
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    sessions.append(session_data)
            except Exception as e:
                print(f"[WARNING] Could not load session {session_file}: {e}")

        return sessions

    def get_learning_metrics(self) -> Dict:
        """
        Calculate aggregated patterns across sessions.

        Returns:
            Dictionary containing:
            - total_sessions: Number of sessions tracked
            - date_range: First to last session dates
            - most_tested_parameters: Parameters tested most frequently
            - most_effective_parameters: Parameters with highest impact
            - recommendation_effectiveness: Success rate of recommendations
            - convergence_metric: Progress toward optimal setup (0-1)
        """
        if not self.metrics_file.exists():
            return {}

        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not load learning metrics: {e}")
            return {}

    def _update_learning_metrics(self, session_data: Dict):
        """
        Update aggregated learning metrics with new session data.

        Args:
            session_data: The session data to incorporate into metrics
        """
        # Load existing metrics or create new
        metrics = self.get_learning_metrics() if self.metrics_file.exists() else {
            "total_sessions": 0,
            "date_range": {"first": None, "last": None},
            "parameter_tests": {},
            "parameter_impacts": {},
            "recommendations": [],
        }

        # Update session count
        metrics["total_sessions"] += 1

        # Update date range
        timestamp = session_data.get("timestamp", datetime.now().isoformat())
        if metrics["date_range"]["first"] is None:
            metrics["date_range"]["first"] = timestamp
        metrics["date_range"]["last"] = timestamp

        # Extract recommended parameter from recommendation string
        recommendation = session_data.get("recommendation", "")
        if recommendation:
            # Parse recommendation to extract parameter name
            # Format: "REDUCE tire_psi_rr by X" or "INCREASE cross_weight by Y"
            param = self._extract_parameter_from_recommendation(recommendation)

            if param:
                # Track parameter test frequency
                metrics["parameter_tests"][param] = metrics["parameter_tests"].get(param, 0) + 1

                # Track parameter impact (from analysis)
                analysis = session_data.get("analysis", {})
                if isinstance(analysis, dict):
                    most_impactful = analysis.get("most_impactful", [])
                    if isinstance(most_impactful, list) and len(most_impactful) >= 2:
                        impact_param, impact_value = most_impactful[0], most_impactful[1]
                        if impact_param == param:
                            if param not in metrics["parameter_impacts"]:
                                metrics["parameter_impacts"][param] = []
                            metrics["parameter_impacts"][param].append(float(impact_value))

                # Store recommendation
                metrics["recommendations"].append({
                    "timestamp": timestamp,
                    "parameter": param,
                    "recommendation": recommendation,
                    "outcome": session_data.get("outcome_feedback")
                })

        # Calculate convergence metric (simplified: based on consistency of recommendations)
        if len(metrics["recommendations"]) >= 3:
            recent_params = [r["parameter"] for r in metrics["recommendations"][-5:]]
            most_common_param = max(set(recent_params), key=recent_params.count)
            convergence = recent_params.count(most_common_param) / len(recent_params)
            metrics["convergence_metric"] = round(convergence, 2)

        # Save updated metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def _extract_parameter_from_recommendation(self, recommendation: str) -> Optional[str]:
        """
        Extract parameter name from recommendation string.

        Args:
            recommendation: Recommendation string (e.g., "REDUCE tire_psi_rr by 0.5 psi")

        Returns:
            Parameter name or None if not found
        """
        # Common parameter names in racing setups
        params = [
            'tire_psi_lf', 'tire_psi_rf', 'tire_psi_lr', 'tire_psi_rr',
            'cross_weight', 'track_bar_height_left', 'track_bar_height_right',
            'spring_lf', 'spring_rf', 'spring_lr', 'spring_rr',
            'arb_front', 'arb_rear', 'toe_lf', 'toe_rf',
            'camber_lf', 'camber_rf', 'camber_lr', 'camber_rr',
            'tire_stagger_lr', 'tire_stagger_rr',
            'spring_ratio_front', 'spring_ratio_rear', 'rake'
        ]

        recommendation_lower = recommendation.lower()
        for param in params:
            if param in recommendation_lower:
                return param

        return None

    def add_outcome_feedback(self, session_id: str, feedback: Dict):
        """
        Record how effective a recommendation was.

        Args:
            session_id: ID of the session to update
            feedback: Dictionary containing outcome data
                - lap_time_improvement: float (seconds)
                - driver_assessment: str ("improved", "no_change", "worse")
                - validated: bool
        """
        session_file = self.storage_dir / f"{session_id}.json"

        if not session_file.exists():
            print(f"[WARNING] Session {session_id} not found")
            return

        # Load session
        with open(session_file, 'r') as f:
            session_data = json.load(f)

        # Add outcome feedback
        session_data["outcome_feedback"] = feedback

        # Save updated session
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        print(f"[SESSION MANAGER] Updated outcome feedback for {session_id}")

    def get_session_summary(self, limit: int = 5) -> str:
        """
        Get a human-readable summary of recent sessions.

        Args:
            limit: Number of recent sessions to summarize

        Returns:
            Formatted string summarizing sessions
        """
        sessions = self.load_session_history(limit)
        metrics = self.get_learning_metrics()

        if not sessions:
            return "No previous sessions found."

        summary = f"\n{'='*60}\n"
        summary += f"SESSION HISTORY ({len(sessions)} recent sessions)\n"
        summary += f"{'='*60}\n\n"

        for i, session in enumerate(sessions, 1):
            timestamp = session.get('timestamp', 'Unknown')
            diagnosis = session.get('driver_diagnosis', {}).get('diagnosis', 'N/A')
            recommendation = session.get('recommendation', 'N/A')

            summary += f"{i}. {timestamp}\n"
            summary += f"   Diagnosis: {diagnosis}\n"
            summary += f"   Recommendation: {recommendation[:80]}...\n"

            outcome = session.get('outcome_feedback')
            if outcome:
                assessment = outcome.get('driver_assessment', 'N/A')
                improvement = outcome.get('lap_time_improvement', 0)
                summary += f"   Outcome: {assessment} ({improvement:+.3f}s)\n"
            summary += "\n"

        # Add learning metrics
        if metrics:
            summary += f"{'='*60}\n"
            summary += "LEARNING METRICS\n"
            summary += f"{'='*60}\n"
            summary += f"Total sessions: {metrics.get('total_sessions', 0)}\n"

            if metrics.get('parameter_tests'):
                most_tested = sorted(
                    metrics['parameter_tests'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                summary += f"Most tested: {', '.join([f'{p} ({c}x)' for p, c in most_tested])}\n"

            if metrics.get('convergence_metric'):
                convergence = metrics['convergence_metric']
                summary += f"Convergence: {convergence:.0%} (focus on consistent parameters)\n"

        return summary
