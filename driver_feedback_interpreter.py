"""
LLM-powered Driver Feedback Interpreter
Converts natural language driver feedback into structured technical diagnosis
"""

import os
import json
from typing import Dict, Optional


def interpret_driver_feedback_with_llm(raw_feedback: str, llm_provider: str = "anthropic") -> Dict:
    """
    Use LLM to interpret natural language driver feedback

    Args:
        raw_feedback: Driver's natural language description (e.g., "Car feels loose in the corners")
        llm_provider: "anthropic", "openai", or "mock" (for testing without API)

    Returns:
        Structured feedback dict with complaint type, priority features, etc.
    """

    prompt = f"""You are a NASCAR crew chief analyzing driver feedback about car handling.

Driver feedback: "{raw_feedback}"

Analyze this feedback and respond with ONLY a JSON object (no markdown, no explanation) with these fields:

{{
    "complaint": "<one of: loose_oversteer, tight_understeer, loose_entry, loose_exit, tight_entry, tight_exit, bottoming, chattering, general>",
    "severity": "<one of: minor, moderate, severe>",
    "phase": "<one of: corner_entry, corner_exit, mid_corner, straightaway, all_phases>",
    "diagnosis": "<concise technical diagnosis in 1 sentence>",
    "priority_features": ["<parameter1>", "<parameter2>", "<parameter3>"]
}}

Priority features should be from this list based on the handling complaint:
- For loose/oversteer: ["tire_psi_rr", "tire_psi_lr", "track_bar_height_left", "spring_rf", "spring_rr"]
- For tight/understeer: ["tire_psi_lf", "tire_psi_rf", "cross_weight", "spring_lf", "spring_rf"]
- For bottoming: ["spring_lf", "spring_rf", "spring_lr", "spring_rr"]
- For general issues: ["tire_psi_rr", "cross_weight", "spring_lf"]

Examples:
Input: "Car is loose off the corners, rear end wants to come around"
Output: {{"complaint": "loose_exit", "severity": "moderate", "phase": "corner_exit", "diagnosis": "Insufficient rear grip causing oversteer on throttle application", "priority_features": ["tire_psi_rr", "tire_psi_lr", "track_bar_height_left"]}}

Input: "Front end pushes in turn 1 and 2"
Output: {{"complaint": "tight_entry", "severity": "moderate", "phase": "corner_entry", "diagnosis": "Insufficient front grip causing understeer at turn-in", "priority_features": ["tire_psi_lf", "tire_psi_rf", "cross_weight"]}}

Now analyze the driver's feedback and return ONLY the JSON object:"""

    if llm_provider == "mock":
        # Mock response for testing without API calls
        return _mock_interpretation(raw_feedback)

    elif llm_provider == "anthropic":
        return _call_anthropic(prompt, raw_feedback)

    elif llm_provider == "openai":
        return _call_openai(prompt, raw_feedback)

    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")


def _call_anthropic(prompt: str, raw_feedback: str) -> Dict:
    """Call Anthropic Claude API"""
    try:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("[WARNING]  ANTHROPIC_API_KEY not found. Using mock interpretation.")
            return _mock_interpretation(raw_feedback)

        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=0.0,  # Deterministic for consistency
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.content[0].text.strip()

        # Parse JSON response
        try:
            result = json.loads(response_text)
            print(f"   [LLM] LLM interpreted feedback successfully")
            return result
        except json.JSONDecodeError:
            print(f"   [WARNING]  LLM response wasn't valid JSON. Using mock interpretation.")
            print(f"   Response was: {response_text[:100]}...")
            return _mock_interpretation(raw_feedback)

    except ImportError:
        print("   [WARNING]  anthropic package not installed. Run: pip install anthropic")
        return _mock_interpretation(raw_feedback)
    except Exception as e:
        print(f"   [WARNING]  LLM call failed: {e}. Using mock interpretation.")
        return _mock_interpretation(raw_feedback)


def _call_openai(prompt: str, raw_feedback: str) -> Dict:
    """Call OpenAI API"""
    try:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[WARNING]  OPENAI_API_KEY not found. Using mock interpretation.")
            return _mock_interpretation(raw_feedback)

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a NASCAR crew chief. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        response_text = response.choices[0].message.content.strip()

        try:
            result = json.loads(response_text)
            print(f"   [LLM] LLM interpreted feedback successfully")
            return result
        except json.JSONDecodeError:
            print(f"   [WARNING]  LLM response wasn't valid JSON. Using mock interpretation.")
            return _mock_interpretation(raw_feedback)

    except ImportError:
        print("   [WARNING]  openai package not installed. Run: pip install openai")
        return _mock_interpretation(raw_feedback)
    except Exception as e:
        print(f"   [WARNING]  LLM call failed: {e}. Using mock interpretation.")
        return _mock_interpretation(raw_feedback)


def _mock_interpretation(raw_feedback: str) -> Dict:
    """
    Mock interpretation using keyword matching
    Fallback when no LLM API is available
    """
    feedback_lower = raw_feedback.lower()

    # Default values
    result = {
        "complaint": "general",
        "severity": "moderate",
        "phase": "all_phases",
        "diagnosis": "General handling optimization needed",
        "priority_features": ["tire_psi_rr", "cross_weight", "spring_lf"]
    }

    # Detect loose/oversteer
    if any(word in feedback_lower for word in ["loose", "oversteer", "rear end", "comes around", "spin"]):
        if "exit" in feedback_lower or "throttle" in feedback_lower or "corner" in feedback_lower:
            result.update({
                "complaint": "loose_exit",
                "phase": "corner_exit",
                "diagnosis": "Insufficient rear grip causing oversteer on throttle application",
                "priority_features": ["tire_psi_rr", "tire_psi_lr", "track_bar_height_left"]
            })
        else:
            result.update({
                "complaint": "loose_oversteer",
                "phase": "mid_corner",
                "diagnosis": "Oversteer (loose rear end) - insufficient rear grip",
                "priority_features": ["tire_psi_rr", "tire_psi_lr", "spring_rf"]
            })

    # Detect tight/understeer
    elif any(word in feedback_lower for word in ["tight", "understeer", "push", "front end", "won't turn"]):
        if "entry" in feedback_lower or "turn in" in feedback_lower:
            result.update({
                "complaint": "tight_entry",
                "phase": "corner_entry",
                "diagnosis": "Insufficient front grip causing understeer at turn-in",
                "priority_features": ["tire_psi_lf", "tire_psi_rf", "cross_weight"]
            })
        else:
            result.update({
                "complaint": "tight_understeer",
                "phase": "mid_corner",
                "diagnosis": "Understeer (tight front end) - insufficient front grip",
                "priority_features": ["tire_psi_lf", "cross_weight", "spring_lf"]
            })

    # Detect bottoming
    elif any(word in feedback_lower for word in ["bottom", "hitting", "harsh", "suspension"]):
        result.update({
            "complaint": "bottoming",
            "phase": "all_phases",
            "diagnosis": "Suspension bottoming - insufficient spring stiffness",
            "priority_features": ["spring_lf", "spring_rf", "spring_lr"]
        })

    # Detect severity
    if any(word in feedback_lower for word in ["really", "very", "extremely", "bad", "terrible"]):
        result["severity"] = "severe"
    elif any(word in feedback_lower for word in ["slight", "little", "bit", "minor"]):
        result["severity"] = "minor"

    print(f"   [FALLBACK] Using rule-based interpretation (no LLM API available)")
    return result


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "Car feels loose coming off the corners, rear end wants to come around",
        "Front end pushes in turn 1 and 2",
        "Car is bottoming out in the center of the corner",
        "A bit tight on entry but loose on exit",
        "Perfect balance, just want to optimize lap time"
    ]

    print("Testing Driver Feedback Interpreter")
    print("=" * 70)

    for feedback in test_cases:
        print(f"\nDriver: \"{feedback}\"")
        result = interpret_driver_feedback_with_llm(feedback, llm_provider="mock")
        print(f"   Complaint: {result['complaint']}")
        print(f"   Severity: {result['severity']}")
        print(f"   Phase: {result['phase']}")
        print(f"   Diagnosis: {result['diagnosis']}")
        print(f"   Priority: {', '.join(result['priority_features'][:3])}")
