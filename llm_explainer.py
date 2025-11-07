"""
Enhanced Agent 3 with LLM-Powered Explanations
Uses Anthropic Claude to generate natural language explanations of decisions
"""

import os
from typing import Dict, Optional


def generate_llm_explanation(
    decision_context: Dict,
    model: str = "claude-3-haiku-20240307"
) -> str:
    """
    Use LLM to generate natural language explanation of Agent 3's decision

    Args:
        decision_context: Dictionary with decision details
        model: Anthropic model to use

    Returns:
        Natural language explanation string
    """

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Fallback to template-based explanation
        return _template_explanation(decision_context)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        # Build prompt for LLM
        prompt = f"""You are an experienced NASCAR crew chief explaining a technical decision to the team.

Context:
- Driver complaint: {decision_context.get('driver_complaint', 'None')}
- Data analysis found: {decision_context.get('data_top_param', 'Unknown')} (correlation: {decision_context.get('data_correlation', 0):.3f})
- Driver's priority areas: {', '.join(decision_context.get('priority_features', []))}
- Decision made: {decision_context.get('decision_type', 'Unknown')}
- Recommended parameter: {decision_context.get('recommended_param', 'Unknown')} (impact: {decision_context.get('recommended_impact', 0):.3f})

Write a brief (2-3 sentences) explanation of why this decision was made. Focus on:
1. Whether data validated or contradicted driver feedback
2. The reasoning behind the recommendation
3. What the team should expect from this change

Be conversational but technical. Use racing terminology."""

        message = client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0.3,  # Slightly creative but consistent
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = message.content[0].text.strip()
        print(f"\n   [LLM] Generated natural language explanation")
        return explanation

    except ImportError:
        print(f"\n   [FALLBACK] anthropic package not installed, using template")
        return _template_explanation(decision_context)
    except Exception as e:
        print(f"\n   [WARNING] LLM explanation failed: {e}, using template")
        return _template_explanation(decision_context)


def _template_explanation(decision_context: Dict) -> str:
    """Fallback template-based explanation"""

    decision_type = decision_context.get('decision_type', 'data_only')
    param = decision_context.get('recommended_param', 'Unknown')
    driver_complaint = decision_context.get('driver_complaint', 'None')

    if decision_type == "driver_validated_by_data":
        return (
            f"The driver reported '{driver_complaint}' and our data analysis confirms this. "
            f"The telemetry shows {param} is the primary factor, validating the driver's intuition. "
            f"This is a high-confidence recommendation backed by both feel and data."
        )
    elif decision_type == "driver_feedback_prioritized":
        return (
            f"While the data suggests a different parameter, we're trusting the driver's feedback about '{driver_complaint}'. "
            f"Drivers feel G-forces and steering response that telemetry can't capture. "
            f"We'll test {param} first, then circle back to the data recommendation if needed."
        )
    elif decision_type == "data_prioritized_no_alternatives":
        return (
            f"The driver reported '{driver_complaint}' but we don't see strong correlations in those areas. "
            f"The data clearly points to {param} as the primary factor. "
            f"We're recommending the data-driven approach while noting the driver's concerns."
        )
    else:
        return (
            f"Based purely on data analysis, {param} shows the strongest correlation with lap time. "
            f"This is a data-driven recommendation to optimize performance."
        )


def generate_llm_multi_turn_analysis(
    session_history: list,
    current_recommendation: Dict,
    model: str = "claude-3-haiku-20240307"
) -> str:
    """
    Advanced feature: Multi-turn LLM analysis that learns from previous sessions

    Args:
        session_history: List of previous session results
        current_recommendation: Current analysis results
        model: Anthropic model to use

    Returns:
        Insights about patterns across sessions
    """

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not session_history:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        # Build context from session history
        history_summary = "\n".join([
            f"Session {i+1}: Driver said '{s.get('driver_complaint', 'N/A')}' → Recommended {s.get('recommended_param', 'N/A')}"
            for i, s in enumerate(session_history[-3:])  # Last 3 sessions
        ])

        prompt = f"""You are analyzing NASCAR testing data across multiple sessions.

Previous sessions:
{history_summary}

Current session:
- Driver complaint: {current_recommendation.get('driver_complaint', 'N/A')}
- Recommendation: {current_recommendation.get('recommended_param', 'N/A')}

Identify any patterns or insights:
1. Are we seeing consistent issues (e.g., always loose, always tight)?
2. Are our changes working or making things worse?
3. Any recommendations for the next test session?

Respond in 2-3 sentences. Be specific and actionable."""

        message = client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        insights = message.content[0].text.strip()
        print(f"\n   [LLM] Generated multi-session insights")
        return insights

    except Exception as e:
        print(f"\n   [WARNING] Multi-turn analysis failed: {e}")
        return None


if __name__ == "__main__":
    # Test the explanation generator
    print("Testing LLM Explanation Generator")
    print("=" * 70)

    # Test case 1: Data validates driver
    context1 = {
        'driver_complaint': 'Oversteer (loose rear end)',
        'data_top_param': 'tire_psi_rr',
        'data_correlation': 0.551,
        'priority_features': ['tire_psi_rr', 'tire_psi_lr'],
        'decision_type': 'driver_validated_by_data',
        'recommended_param': 'tire_psi_rr',
        'recommended_impact': 0.551
    }

    print("\nScenario 1: Data validates driver")
    print("-" * 70)
    explanation1 = generate_llm_explanation(context1)
    print(explanation1)

    # Test case 2: Data contradicts driver
    context2 = {
        'driver_complaint': 'Understeer (tight front end)',
        'data_top_param': 'tire_psi_rr',
        'data_correlation': 0.551,
        'priority_features': ['tire_psi_lf', 'cross_weight'],
        'decision_type': 'driver_feedback_prioritized',
        'recommended_param': 'cross_weight',
        'recommended_impact': -0.289
    }

    print("\n\nScenario 2: Data contradicts driver")
    print("-" * 70)
    explanation2 = generate_llm_explanation(context2)
    print(explanation2)
