"""
Intelligent Input Router for AI Race Engineer
Analyzes user input to determine which features and analysis modes to activate
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DriverFeedback:
    """Parsed driver feedback from natural language input"""
    complaint: str  # 'loose_oversteer', 'tight_understeer', 'bottoming', etc.
    description: str
    severity: str  # 'minor', 'moderate', 'severe'
    phase: str  # 'corner_entry', 'mid_corner', 'corner_exit', 'straight'
    raw_input: str


@dataclass
class AnalysisRequest:
    """Parsed analysis request from user input"""
    driver_feedback: Optional[DriverFeedback]
    analysis_type: str  # 'setup_optimization', 'performance_comparison', 'diagnostic'
    focus_areas: List[str]  # e.g., ['rear_grip', 'aero_balance', 'tire_pressure']
    verbosity: str  # 'concise', 'detailed'


class InputRouter:
    """Routes user input to appropriate analysis features"""

    # Pattern matching for driver complaints
    COMPLAINT_PATTERNS = {
        'loose_oversteer': [
            r'loose', r'oversteer', r'rear.*slide', r'tail.*out',
            r'back.*end.*come.*around', r'snap.*oversteer',
            r'rear.*grip', r'unstable.*rear'
        ],
        'tight_understeer': [
            r'tight', r'understeer', r'push', r'plow',
            r'front.*wash.*out', r'won\'t.*turn', r'front.*grip'
        ],
        'bottoming': [
            r'bottom', r'harsh', r'hit.*ground', r'stiff',
            r'ride.*height', r'porpois'
        ],
        'poor_traction': [
            r'traction', r'wheel.*spin', r'throttle.*application',
            r'drive.*off.*corner', r'power.*down'
        ],
        'brake_balance': [
            r'brake', r'lock.*up', r'stopping.*power',
            r'front.*rear.*brake'
        ],
        'aero_balance': [
            r'aero', r'downforce', r'high.*speed.*corner',
            r'wing', r'splitter'
        ]
    }

    # Severity indicators
    SEVERITY_PATTERNS = {
        'severe': [r'really', r'very', r'extremely', r'terrible', r'awful', r'major'],
        'moderate': [r'bit', r'somewhat', r'little', r'slight'],
        'minor': [r'tiny', r'barely', r'hardly']
    }

    # Phase indicators
    PHASE_PATTERNS = {
        'corner_entry': [r'turn.*in', r'entry', r'braking.*zone', r'corner.*entry'],
        'mid_corner': [r'mid.*corner', r'apex', r'middle.*corner'],
        'corner_exit': [r'exit', r'off.*corner', r'throttle.*application', r'drive.*off'],
        'straight': [r'straight', r'full.*throttle', r'down.*straight']
    }

    def parse_input(self, user_input: str) -> AnalysisRequest:
        """
        Parse user input to determine analysis requirements

        Args:
            user_input: Natural language input from user

        Returns:
            AnalysisRequest with parsed components
        """
        user_input_lower = user_input.lower()

        # 1. Detect driver feedback
        driver_feedback = self._extract_driver_feedback(user_input, user_input_lower)

        # 2. Determine analysis type
        analysis_type = self._determine_analysis_type(user_input_lower, driver_feedback)

        # 3. Identify focus areas
        focus_areas = self._identify_focus_areas(user_input_lower, driver_feedback)

        # 4. Determine verbosity preference
        verbosity = self._determine_verbosity(user_input_lower)

        return AnalysisRequest(
            driver_feedback=driver_feedback,
            analysis_type=analysis_type,
            focus_areas=focus_areas,
            verbosity=verbosity
        )

    def _extract_driver_feedback(self, raw_input: str, input_lower: str) -> Optional[DriverFeedback]:
        """Extract driver feedback if present in input"""

        # Look for driver feedback indicators OR complaint keywords
        feedback_indicators = [
            r'car.*feel', r'driver.*say', r'feedback', r'complain',
            r'handling', r'issue', r'problem'
        ]

        # Check if any complaint patterns match directly (even without explicit indicators)
        has_complaint = False
        for patterns in self.COMPLAINT_PATTERNS.values():
            if any(re.search(pattern, input_lower) for pattern in patterns):
                has_complaint = True
                break

        has_feedback = any(re.search(pattern, input_lower) for pattern in feedback_indicators)

        # Must have either feedback indicators OR direct complaints
        if not has_feedback and not has_complaint:
            return None

        # Identify complaint type
        complaint = None
        for complaint_type, patterns in self.COMPLAINT_PATTERNS.items():
            if any(re.search(pattern, input_lower) for pattern in patterns):
                complaint = complaint_type
                break

        if not complaint:
            complaint = 'general_handling'  # Default

        # Determine severity
        severity = 'moderate'  # Default
        for sev_level, patterns in self.SEVERITY_PATTERNS.items():
            if any(re.search(pattern, input_lower) for pattern in patterns):
                severity = sev_level
                break

        # Determine phase
        phase = 'general'  # Default
        for phase_type, patterns in self.PHASE_PATTERNS.items():
            if any(re.search(pattern, input_lower) for pattern in patterns):
                phase = phase_type
                break

        return DriverFeedback(
            complaint=complaint,
            description=raw_input,
            severity=severity,
            phase=phase,
            raw_input=raw_input
        )

    def _determine_analysis_type(self, input_lower: str,
                                 driver_feedback: Optional[DriverFeedback]) -> str:
        """Determine what type of analysis is needed"""

        # Check for explicit analysis type requests
        if re.search(r'compar', input_lower):
            return 'performance_comparison'

        if re.search(r'diagnos|what.*wrong|problem|issue', input_lower):
            return 'diagnostic'

        # If driver feedback present, assume setup optimization
        if driver_feedback:
            return 'setup_optimization'

        # Default to setup optimization
        return 'setup_optimization'

    def _identify_focus_areas(self, input_lower: str,
                             driver_feedback: Optional[DriverFeedback]) -> List[str]:
        """Identify specific areas to focus analysis on"""

        focus_areas = []

        # Map complaints to focus areas
        complaint_focus = {
            'loose_oversteer': ['rear_grip', 'tire_pressure_rear', 'rear_springs'],
            'tight_understeer': ['front_grip', 'tire_pressure_front', 'front_springs'],
            'bottoming': ['ride_height', 'springs', 'dampers'],
            'poor_traction': ['rear_grip', 'differential', 'tire_pressure'],
            'brake_balance': ['brake_bias', 'brake_pressure'],
            'aero_balance': ['wing_angle', 'aero_balance', 'ride_height']
        }

        # Add focus areas from driver feedback
        if driver_feedback and driver_feedback.complaint in complaint_focus:
            focus_areas.extend(complaint_focus[driver_feedback.complaint])

        # Look for explicit parameter mentions
        parameter_keywords = {
            'tire': ['tire_pressure', 'tire_temps'],
            'spring': ['springs', 'ride_height'],
            'damper': ['dampers', 'bump', 'rebound'],
            'aero': ['wing_angle', 'aero_balance'],
            'brake': ['brake_bias', 'brake_pressure'],
            'diff': ['differential']
        }

        for keyword, params in parameter_keywords.items():
            if keyword in input_lower:
                focus_areas.extend(params)

        # Remove duplicates
        focus_areas = list(set(focus_areas))

        return focus_areas if focus_areas else ['general']

    def _determine_verbosity(self, input_lower: str) -> str:
        """Determine preferred output verbosity"""

        verbose_indicators = [
            r'detail', r'explain', r'show.*all', r'verbose',
            r'step.*by.*step', r'how.*work'
        ]

        concise_indicators = [
            r'quick', r'brief', r'summary', r'just.*tell',
            r'bottom.*line', r'short'
        ]

        if any(re.search(pattern, input_lower) for pattern in verbose_indicators):
            return 'detailed'

        if any(re.search(pattern, input_lower) for pattern in concise_indicators):
            return 'concise'

        # Default to concise for better UX
        return 'concise'

    def create_driver_feedback_dict(self, feedback: DriverFeedback) -> Dict:
        """Convert DriverFeedback to dict format expected by race engineer"""
        return {
            'complaint': feedback.complaint,
            'description': feedback.description,
            'severity': feedback.severity,
            'phase': feedback.phase
        }


# Example usage
if __name__ == '__main__':
    router = InputRouter()

    # Test cases
    test_inputs = [
        "The car feels really loose coming off corners, especially turns 1 and 2",
        "Car is pushing in the entry, can't get it to turn in",
        "We're bottoming out in turn 3, feels very harsh",
        "Quick analysis of lap times",
        "Show me detailed breakdown of tire pressure effects"
    ]

    print("Input Router Test Cases")
    print("=" * 60)

    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n{i}. Input: \"{test_input}\"")
        request = router.parse_input(test_input)

        print(f"   Analysis Type: {request.analysis_type}")
        print(f"   Focus Areas: {', '.join(request.focus_areas)}")
        print(f"   Verbosity: {request.verbosity}")

        if request.driver_feedback:
            fb = request.driver_feedback
            print(f"   Driver Feedback:")
            print(f"     - Complaint: {fb.complaint}")
            print(f"     - Severity: {fb.severity}")
            print(f"     - Phase: {fb.phase}")
        else:
            print(f"   Driver Feedback: None detected")
