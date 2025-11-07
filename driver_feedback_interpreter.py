"""
LLM-powered Driver Feedback Interpreter
Converts natural language driver feedback into structured technical diagnosis
Supports MULTIPLE concurrent issues in a single feedback input
"""

import os
import json
from typing import Dict, Optional, List


def interpret_driver_feedback_with_llm(raw_feedback: str, llm_provider: str = "anthropic", multi_issue: bool = True) -> Dict:
    """
    Use LLM to interpret natural language driver feedback (supports multiple concurrent issues)

    Args:
        raw_feedback: Driver's natural language description
        llm_provider: "anthropic", "openai", or "mock" (for testing without API)
        multi_issue: If True, parse multiple issues; if False, use legacy single-issue parsing

    Returns:
        If multi_issue=True:
            {
                'issues': [list of issue dicts],
                'primary_issue': most severe issue,
                'handling_balance_type': 'understeer_dominant' | 'oversteer_dominant' | 'mixed' | 'balanced'
            }
        If multi_issue=False:
            Legacy single-issue format (for backward compatibility)
    """

    if not multi_issue:
        # Legacy single-issue parsing
        return _interpret_single_issue(raw_feedback, llm_provider)

    # Multi-issue parsing
    prompt = f"""You are a NASCAR crew chief analyzing driver feedback about car handling.

Driver feedback: "{raw_feedback}"

The driver may describe MULTIPLE handling issues (e.g., "loose on exit but tight on entry").
Parse ALL distinct issues mentioned.

Respond with ONLY a JSON object (no markdown, no explanation):

{{
    "issues": [
        {{
            "complaint": "<loose_exit|tight_entry|loose_entry|tight_exit|loose_oversteer|tight_understeer|bottoming|chattering|general>",
            "severity": "<minor|moderate|severe>",
            "phase": "<corner_entry|corner_exit|mid_corner|straightaway|all_phases>",
            "diagnosis": "<1 sentence technical diagnosis>",
            "priority_features": ["param1", "param2", "param3"],
            "confidence": 0.95
        }}
    ],
    "handling_balance_type": "<understeer_dominant|oversteer_dominant|mixed|balanced|optimization>",
    "primary_issue_index": 0
}}

Rules:
1. Create separate issue objects for each DISTINCT handling problem
2. If "but", "however", "also", "and" appears with different complaints, likely multiple issues
3. Rate confidence 0-1 based on specificity of driver description
4. primary_issue_index points to most severe/impactful issue (highest severity)
5. If only optimization requested, use "optimization" for balance_type and empty issues array

Priority features by complaint type:
- loose_exit/loose_oversteer: ["tire_psi_rr", "tire_psi_lr", "track_bar_height_left"]
- tight_entry/tight_understeer: ["tire_psi_lf", "tire_psi_rf", "cross_weight"]
- bottoming: ["spring_lf", "spring_rf", "spring_lr", "spring_rr"]

Examples:

Input: "Car is loose off corners but tight on entry"
Output: {{"issues": [{{"complaint": "loose_exit", "severity": "moderate", "phase": "corner_exit", "diagnosis": "Insufficient rear grip on throttle", "priority_features": ["tire_psi_rr", "tire_psi_lr", "track_bar_height_left"], "confidence": 0.9}}, {{"complaint": "tight_entry", "severity": "moderate", "phase": "corner_entry", "diagnosis": "Insufficient front grip at turn-in", "priority_features": ["tire_psi_lf", "tire_psi_rf", "cross_weight"], "confidence": 0.85}}], "handling_balance_type": "mixed", "primary_issue_index": 0}}

Input: "Bottoming in turn 2 and loose everywhere"
Output: {{"issues": [{{"complaint": "bottoming", "severity": "severe", "phase": "mid_corner", "diagnosis": "Suspension bottoming limits platform control", "priority_features": ["spring_lf", "spring_rf", "spring_lr"], "confidence": 0.95}}, {{"complaint": "loose_oversteer", "severity": "moderate", "phase": "all_phases", "diagnosis": "General oversteer throughout corner", "priority_features": ["tire_psi_rr", "tire_psi_lr"], "confidence": 0.8}}], "handling_balance_type": "oversteer_dominant", "primary_issue_index": 0}}

Input: "Perfect balance, just want a few tenths"
Output: {{"issues": [], "handling_balance_type": "optimization", "primary_issue_index": null}}

Now analyze the driver's feedback and return ONLY the JSON object:"""

    if llm_provider == "mock":
        # Mock response for testing without API calls
        return _mock_multi_issue_interpretation(raw_feedback)

    elif llm_provider == "anthropic":
        result = _call_anthropic(prompt, raw_feedback)
        # Post-process to extract primary issue
        if result and 'issues' in result:
            primary_idx = result.get('primary_issue_index', 0)
            if result['issues'] and primary_idx is not None and primary_idx < len(result['issues']):
                result['primary_issue'] = result['issues'][primary_idx]
            else:
                result['primary_issue'] = result['issues'][0] if result['issues'] else None
        return result

    elif llm_provider == "openai":
        result = _call_openai(prompt, raw_feedback)
        # Post-process to extract primary issue
        if result and 'issues' in result:
            primary_idx = result.get('primary_issue_index', 0)
            if result['issues'] and primary_idx is not None and primary_idx < len(result['issues']):
                result['primary_issue'] = result['issues'][primary_idx]
            else:
                result['primary_issue'] = result['issues'][0] if result['issues'] else None
        return result

    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")


def _interpret_single_issue(raw_feedback: str, llm_provider: str) -> Dict:
    """
    Legacy single-issue interpretation (backward compatibility)
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
        return _mock_interpretation(raw_feedback)
    elif llm_provider == "anthropic":
        return _call_anthropic(prompt, raw_feedback)
    elif llm_provider == "openai":
        return _call_openai(prompt, raw_feedback)
    else:
        return _mock_interpretation(raw_feedback)


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
            model="claude-3-haiku-20240307",
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


def _mock_multi_issue_interpretation(raw_feedback: str) -> Dict:
    """
    Mock multi-issue interpretation using keyword matching
    Fallback when no LLM API is available
    """
    feedback_lower = raw_feedback.lower()
    issues = []

    # Check for optimization mode
    if any(word in feedback_lower for word in ["perfect", "balanced", "good", "optimize", "fine tune", "few tenths"]):
        if not any(word in feedback_lower for word in ["loose", "tight", "bottom", "push", "oversteer", "understeer"]):
            print(f"   [FALLBACK] Multi-issue rule-based interpretation (optimization mode)")
            return {
                "issues": [],
                "handling_balance_type": "optimization",
                "primary_issue_index": None,
                "primary_issue": None
            }

    # Detect loose/oversteer issues
    loose_detected = False
    if any(word in feedback_lower for word in ["loose", "oversteer", "rear end", "comes around", "spin"]):
        if "exit" in feedback_lower or "throttle" in feedback_lower:
            issues.append({
                "complaint": "loose_exit",
                "severity": "moderate",
                "phase": "corner_exit",
                "diagnosis": "Insufficient rear grip causing oversteer on throttle application",
                "priority_features": ["tire_psi_rr", "tire_psi_lr", "track_bar_height_left"],
                "confidence": 0.85
            })
            loose_detected = True
        elif "entry" in feedback_lower:
            issues.append({
                "complaint": "loose_entry",
                "severity": "moderate",
                "phase": "corner_entry",
                "diagnosis": "Oversteer on turn-in",
                "priority_features": ["tire_psi_rr", "cross_weight"],
                "confidence": 0.80
            })
            loose_detected = True
        elif not loose_detected:
            issues.append({
                "complaint": "loose_oversteer",
                "severity": "moderate",
                "phase": "all_phases",
                "diagnosis": "General oversteer - insufficient rear grip",
                "priority_features": ["tire_psi_rr", "tire_psi_lr", "spring_rf"],
                "confidence": 0.85
            })
            loose_detected = True

    # Detect tight/understeer issues
    tight_detected = False
    if any(word in feedback_lower for word in ["tight", "understeer", "push", "front end", "won't turn"]):
        if "entry" in feedback_lower or "turn in" in feedback_lower:
            issues.append({
                "complaint": "tight_entry",
                "severity": "moderate",
                "phase": "corner_entry",
                "diagnosis": "Insufficient front grip causing understeer at turn-in",
                "priority_features": ["tire_psi_lf", "tire_psi_rf", "cross_weight"],
                "confidence": 0.85
            })
            tight_detected = True
        elif "exit" in feedback_lower:
            issues.append({
                "complaint": "tight_exit",
                "severity": "moderate",
                "phase": "corner_exit",
                "diagnosis": "Understeer on corner exit",
                "priority_features": ["tire_psi_lf", "cross_weight"],
                "confidence": 0.80
            })
            tight_detected = True
        elif not tight_detected:
            issues.append({
                "complaint": "tight_understeer",
                "severity": "moderate",
                "phase": "mid_corner",
                "diagnosis": "General understeer - insufficient front grip",
                "priority_features": ["tire_psi_lf", "tire_psi_rf", "cross_weight"],
                "confidence": 0.85
            })
            tight_detected = True

    # Detect bottoming
    if any(word in feedback_lower for word in ["bottom", "hitting", "harsh", "suspension"]):
        issues.append({
            "complaint": "bottoming",
            "severity": "severe",
            "phase": "all_phases",
            "diagnosis": "Suspension bottoming - insufficient spring stiffness",
            "priority_features": ["spring_lf", "spring_rf", "spring_lr", "spring_rr"],
            "confidence": 0.90
        })

    # Detect chattering
    if any(word in feedback_lower for word in ["chatter", "vibrat", "bouncing", "hopping"]):
        issues.append({
            "complaint": "chattering",
            "severity": "moderate",
            "phase": "mid_corner",
            "diagnosis": "Tire chattering - possible over-stiff springs or excessive load",
            "priority_features": ["spring_lf", "spring_rf", "tire_psi_lf"],
            "confidence": 0.75
        })

    # Adjust severity based on keywords
    for issue in issues:
        if any(word in feedback_lower for word in ["really", "very", "extremely", "bad", "terrible", "severe"]):
            issue["severity"] = "severe"
        elif any(word in feedback_lower for word in ["slight", "little", "bit", "minor", "small"]):
            issue["severity"] = "minor"

    # Determine handling balance type
    has_loose = any(i['complaint'] in ['loose_exit', 'loose_entry', 'loose_oversteer'] for i in issues)
    has_tight = any(i['complaint'] in ['tight_exit', 'tight_entry', 'tight_understeer'] for i in issues)

    if has_loose and has_tight:
        balance_type = "mixed"
    elif has_loose:
        balance_type = "oversteer_dominant"
    elif has_tight:
        balance_type = "understeer_dominant"
    elif issues:
        balance_type = "balanced"
    else:
        balance_type = "general"

    # If no specific issues detected, add general issue
    if not issues:
        issues.append({
            "complaint": "general",
            "severity": "moderate",
            "phase": "all_phases",
            "diagnosis": "General handling optimization needed",
            "priority_features": ["tire_psi_rr", "cross_weight", "spring_lf"],
            "confidence": 0.70
        })
        balance_type = "balanced"

    # Determine primary issue (highest severity, then confidence)
    severity_order = {"severe": 3, "moderate": 2, "minor": 1}
    primary_idx = 0
    max_priority = (severity_order.get(issues[0]['severity'], 0), issues[0]['confidence'])

    for i, issue in enumerate(issues):
        priority = (severity_order.get(issue['severity'], 0), issue['confidence'])
        if priority > max_priority:
            max_priority = priority
            primary_idx = i

    print(f"   [FALLBACK] Multi-issue rule-based interpretation ({len(issues)} issue(s) detected)")
    return {
        "issues": issues,
        "handling_balance_type": balance_type,
        "primary_issue_index": primary_idx,
        "primary_issue": issues[primary_idx] if issues else None
    }


def _mock_interpretation(raw_feedback: str) -> Dict:
    """
    Mock interpretation using keyword matching (single-issue legacy)
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
