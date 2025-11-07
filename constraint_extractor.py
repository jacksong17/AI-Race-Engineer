"""
Constraint Extractor for Driver Feedback

Parses driver feedback to identify:
- Parameter limits (e.g., "RR tire PSI is as low as legally allowed")
- Already-tried changes (e.g., "we already increased cross weight")
- Physical constraints (e.g., "can't go any softer on springs")
"""

import re
from typing import Dict, List, Optional


def extract_constraints(raw_feedback: str, llm_provider: str = "anthropic") -> Dict:
    """
    Extract constraints and limits from driver feedback.

    Args:
        raw_feedback: Raw driver feedback string
        llm_provider: LLM provider for constraint extraction

    Returns:
        Dict with:
            - parameter_limits: List of {param, limit_type, reason}
            - already_tried: List of parameters already adjusted
            - cannot_adjust: List of parameters that cannot be changed
            - raw_constraints: Original constraint text
    """

    # Try LLM extraction first (most accurate)
    if llm_provider == "anthropic":
        llm_result = _extract_constraints_with_llm(raw_feedback)
        if llm_result:
            return llm_result

    # Fallback to rule-based extraction
    return _extract_constraints_rule_based(raw_feedback)


def _extract_constraints_with_llm(raw_feedback: str) -> Optional[Dict]:
    """
    Use LLM to extract constraints from feedback.

    Returns None if LLM unavailable.
    """
    try:
        import anthropic
        import os
        import json

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""Analyze this driver feedback for EXPLICIT setup constraints and limits:

"{raw_feedback}"

IMPORTANT: Only extract constraints if EXPLICITLY stated by the driver. Do NOT infer or assume constraints.

Extract ONLY if driver explicitly mentions:
1. Parameters at their limit (e.g., "tire pressure as low as allowed", "can't go any lower on RR tire")
2. Parameters already adjusted (e.g., "we already tried increasing springs", "already raised cross weight last run")
3. Parameters that cannot be changed (e.g., "can't adjust track bar", "springs are locked in")

Respond with ONLY a JSON object:

{{
    "parameter_limits": [
        {{"param": "tire_psi_rr", "limit_type": "at_minimum", "reason": "as low as legally allowed"}},
        ...
    ],
    "already_tried": ["cross_weight", ...],
    "cannot_adjust": ["spring_lf", ...],
    "raw_constraints": ["exact text mentioning constraint", ...]
}}

Parameter names must be from:
tire_psi_lf, tire_psi_rf, tire_psi_lr, tire_psi_rr, cross_weight, track_bar_height_left,
spring_lf, spring_rf, spring_lr, spring_rr, arb_front, arb_rear

Limit types: "at_minimum", "at_maximum", "near_limit", "optimal"

If NO EXPLICIT constraints mentioned, return empty arrays for all fields.

Examples:
- "Car feels loose" → {{"parameter_limits": [], "already_tried": [], "cannot_adjust": [], "raw_constraints": []}}
- "Car feels bad, turtles drive faster" → {{"parameter_limits": [], "already_tried": [], "cannot_adjust": [], "raw_constraints": []}}
- "RR tire is as low as we can go, still loose" → {{"parameter_limits": [{{"param": "tire_psi_rr", "limit_type": "at_minimum", "reason": "as low as we can go"}}], "already_tried": [], "cannot_adjust": [], "raw_constraints": ["RR tire is as low as we can go"]}}

Return ONLY the JSON:"""

        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        try:
            result = json.loads(response_text)
            print(f"   [CONSTRAINTS] LLM extracted {len(result.get('parameter_limits', []))} constraint(s)")
            return result
        except json.JSONDecodeError:
            print(f"   [WARNING] LLM constraint extraction failed, using rule-based")
            return None

    except ImportError:
        return None
    except Exception as e:
        print(f"   [WARNING] Constraint extraction error: {str(e)[:50]}")
        return None


def _extract_constraints_rule_based(raw_feedback: str) -> Dict:
    """
    Rule-based constraint extraction (fallback).

    Looks for patterns like:
    - "as low as allowed", "at minimum", "maxed out"
    - "already tried", "already increased/decreased"
    - "can't adjust", "cannot change"
    """

    feedback_lower = raw_feedback.lower()

    parameter_limits = []
    already_tried = []
    cannot_adjust = []
    raw_constraints = []

    # Define parameter patterns
    param_patterns = {
        'tire_psi_rr': r'\b(rr|right rear|rear right)\s*(tire\s*)?(psi|pressure)',
        'tire_psi_rl': r'\b(lr|left rear|rear left)\s*(tire\s*)?(psi|pressure)',
        'tire_psi_rf': r'\b(rf|right front|front right)\s*(tire\s*)?(psi|pressure)',
        'tire_psi_lf': r'\b(lf|left front|front left)\s*(tire\s*)?(psi|pressure)',
        'cross_weight': r'\b(cross\s*weight|wedge)',
        'spring_lf': r'\b(lf|left front)\s*spring',
        'spring_rf': r'\b(rf|right front)\s*spring',
        'spring_lr': r'\b(lr|left rear)\s*spring',
        'spring_rr': r'\b(rr|right rear)\s*spring',
        'track_bar': r'\b(track\s*bar|panhard)',
    }

    # Limit patterns
    limit_patterns = {
        'at_minimum': [
            r'as low as (?:legally )?(?:allowed|possible|permitted)',
            r'at (?:the )?minimum',
            r'can\'?t go (?:any )?lower',
            r'minimum (?:legal )?(?:limit|value)',
            r'lowest (?:we can|possible|allowed)',
        ],
        'at_maximum': [
            r'as high as (?:legally )?(?:allowed|possible|permitted)',
            r'at (?:the )?maximum',
            r'maxed out',
            r'can\'?t go (?:any )?higher',
            r'maximum (?:legal )?(?:limit|value)',
        ],
        'near_limit': [
            r'(?:almost|nearly|close to) (?:the )?(?:limit|maximum|minimum)',
            r'not much (?:room|range) left',
        ]
    }

    # Already tried patterns
    tried_patterns = [
        r'(?:already|previously) (?:tried|tested|adjusted|changed|increased|decreased)',
        r'(?:tried|tested) (?:increasing|decreasing|adjusting|changing)',
        r'we (?:tried|tested|already)',
    ]

    # Cannot adjust patterns
    cannot_patterns = [
        r'can\'?t (?:adjust|change|modify)',
        r'cannot (?:adjust|change|modify)',
        r'unable to (?:adjust|change|modify)',
        r'locked (?:in|down)',
    ]

    # Extract parameter limits
    for param, param_pattern in param_patterns.items():
        param_match = re.search(param_pattern, feedback_lower)
        if param_match:
            # Check for limit indicators near this parameter
            # Look in a 50-character window around the match
            start = max(0, param_match.start() - 50)
            end = min(len(feedback_lower), param_match.end() + 50)
            context = feedback_lower[start:end]

            for limit_type, patterns in limit_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, context):
                        # Extract the original text
                        original_text = raw_feedback[start:end].strip()
                        reason = re.search(pattern, context).group(0)

                        parameter_limits.append({
                            'param': param,
                            'limit_type': limit_type,
                            'reason': reason
                        })
                        raw_constraints.append(original_text)
                        break

    # Extract already tried
    for pattern in tried_patterns:
        match = re.search(pattern, feedback_lower)
        if match:
            # Look for parameter names near this mention
            start = max(0, match.start() - 20)
            end = min(len(feedback_lower), match.end() + 50)
            context = feedback_lower[start:end]

            for param, param_pattern in param_patterns.items():
                if re.search(param_pattern, context):
                    already_tried.append(param)
                    raw_constraints.append(raw_feedback[start:end].strip())

    # Extract cannot adjust
    for pattern in cannot_patterns:
        match = re.search(pattern, feedback_lower)
        if match:
            start = max(0, match.start() - 20)
            end = min(len(feedback_lower), match.end() + 50)
            context = feedback_lower[start:end]

            for param, param_pattern in param_patterns.items():
                if re.search(param_pattern, context):
                    cannot_adjust.append(param)
                    raw_constraints.append(raw_feedback[start:end].strip())

    result = {
        'parameter_limits': parameter_limits,
        'already_tried': list(set(already_tried)),
        'cannot_adjust': list(set(cannot_adjust)),
        'raw_constraints': list(set(raw_constraints))
    }

    if parameter_limits or already_tried or cannot_adjust:
        print(f"   [CONSTRAINTS] Rule-based extracted {len(parameter_limits)} limit(s), "
              f"{len(already_tried)} tried, {len(cannot_adjust)} locked")

    return result


def get_constraint_summary(constraints: Dict) -> str:
    """
    Get human-readable summary of constraints.

    Args:
        constraints: Output from extract_constraints()

    Returns:
        Formatted string summarizing constraints
    """
    if not any([
        constraints.get('parameter_limits'),
        constraints.get('already_tried'),
        constraints.get('cannot_adjust')
    ]):
        return "No constraints identified"

    summary_parts = []

    # Parameter limits
    if constraints.get('parameter_limits'):
        summary_parts.append("**Parameter Limits:**")
        for limit in constraints['parameter_limits']:
            summary_parts.append(f"  - {limit['param']}: {limit['limit_type']} ({limit['reason']})")

    # Already tried
    if constraints.get('already_tried'):
        summary_parts.append("**Already Tried:**")
        summary_parts.append(f"  - {', '.join(constraints['already_tried'])}")

    # Cannot adjust
    if constraints.get('cannot_adjust'):
        summary_parts.append("**Cannot Adjust:**")
        summary_parts.append(f"  - {', '.join(constraints['cannot_adjust'])}")

    return "\n".join(summary_parts)


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "Car feels loose off corners. RR tire PSI is as low as legally allowed currently.",
        "Front end pushes. We already tried increasing cross weight but it didn't help.",
        "Bottoming out in turn 2. Springs are maxed out, can't go any stiffer.",
        "Loose on entry, tight on exit. Already tested rear tire pressure changes.",
    ]

    print("Testing constraint extraction:\n")
    for i, feedback in enumerate(test_cases, 1):
        print(f"Test {i}: {feedback}")
        constraints = extract_constraints(feedback, llm_provider="mock")
        print(get_constraint_summary(constraints))
        print()
