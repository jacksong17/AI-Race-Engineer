"""
Knowledge Base Loader for Car Setup Manuals

Loads and indexes car setup documentation (PDFs) to provide:
- Legal parameter limits
- Setup guidance for specific handling issues
- Parameter interaction information
- Best practices and recommendations
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_setup_manual(pdf_path: str = None) -> Dict:
    """
    Load car setup manual from PDF.

    Args:
        pdf_path: Path to iRacing setup manual PDF (default: ./docs/setup_manual.pdf)

    Returns:
        Dict with:
            - parameter_limits: Legal ranges for each parameter
            - handling_guides: Recommendations by handling issue
            - parameter_interactions: Known synergies and conflicts
            - indexed_content: Searchable manual sections
    """

    if pdf_path is None:
        pdf_path = Path("docs/setup_manual.pdf")
    else:
        pdf_path = Path(pdf_path)

    knowledge_base = {
        'parameter_limits': {},
        'handling_guides': {},
        'parameter_interactions': {},
        'indexed_content': {},
        'source': None
    }

    # Try to load PDF if available
    if pdf_path.exists():
        print(f"   [KNOWLEDGE] Loading setup manual from {pdf_path}")
        try:
            knowledge_base = _load_pdf_manual(pdf_path)
            knowledge_base['source'] = str(pdf_path)
            print(f"   [OK] Loaded {len(knowledge_base.get('indexed_content', {}))} manual sections")
            return knowledge_base
        except Exception as e:
            print(f"   [WARNING] Could not load PDF: {e}")

    # Fallback to default NASCAR knowledge
    print(f"   [KNOWLEDGE] Using default NASCAR setup knowledge")
    knowledge_base = _load_default_nascar_knowledge()
    knowledge_base['source'] = "default"

    return knowledge_base


def _load_pdf_manual(pdf_path: Path) -> Dict:
    """
    Extract and index content from PDF setup manual.

    Requires: pip install PyPDF2
    """
    try:
        import PyPDF2
    except ImportError:
        print("   [INFO] PyPDF2 not installed. Run: pip install PyPDF2")
        print("   [INFO] Using default knowledge base instead")
        return _load_default_nascar_knowledge()

    knowledge_base = {
        'parameter_limits': {},
        'handling_guides': {},
        'parameter_interactions': {},
        'indexed_content': {}
    }

    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            full_text = []
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                full_text.append(text)

                # Index this page
                knowledge_base['indexed_content'][f'page_{page_num + 1}'] = text

            # Extract parameter limits from text
            combined_text = '\n'.join(full_text)
            knowledge_base['parameter_limits'] = _extract_parameter_limits(combined_text)

            # Extract handling guides
            knowledge_base['handling_guides'] = _extract_handling_guides(combined_text)

            # Extract parameter interactions
            knowledge_base['parameter_interactions'] = _extract_parameter_interactions(combined_text)

    except Exception as e:
        print(f"   [WARNING] Error reading PDF: {e}")
        return _load_default_nascar_knowledge()

    return knowledge_base


def _extract_parameter_limits(text: str) -> Dict:
    """
    Extract legal parameter ranges from manual text.

    Looks for patterns like:
    - "Tire pressure: 18-35 psi"
    - "Cross weight range: 48%-56%"
    - "Minimum tire pressure: 18 psi"
    """
    import re

    limits = {}

    # Common parameter limit patterns
    patterns = {
        'tire_psi': [
            r'tire\s+pressure[:\s]+(\d+)\s*-\s*(\d+)\s*psi',
            r'minimum\s+tire\s+pressure[:\s]+(\d+)\s*psi',
            r'maximum\s+tire\s+pressure[:\s]+(\d+)\s*psi',
        ],
        'cross_weight': [
            r'cross\s*weight[:\s]+(\d+\.?\d*)%?\s*-\s*(\d+\.?\d*)%',
            r'wedge[:\s]+(\d+\.?\d*)%?\s*-\s*(\d+\.?\d*)%',
        ],
    }

    for param_key, param_patterns in patterns.items():
        for pattern in param_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if len(matches[0]) == 2:  # Range found
                    limits[param_key] = {
                        'min': float(matches[0][0]),
                        'max': float(matches[0][1]),
                        'unit': 'psi' if 'psi' in param_key else '%'
                    }
                    break

    return limits


def _extract_handling_guides(text: str) -> Dict:
    """
    Extract handling issue recommendations from manual.

    Looks for sections like:
    - "For loose on exit, try..."
    - "Understeer can be corrected by..."
    """
    import re

    guides = {}

    # Common handling issue patterns
    issue_patterns = {
        'loose_exit': [r'loose\s+(?:on\s+)?exit', r'oversteer\s+(?:on\s+)?exit'],
        'loose_entry': [r'loose\s+(?:on\s+)?entry', r'oversteer\s+(?:on\s+)?entry'],
        'tight_understeer': [r'tight', r'understeer', r'push'],
        'bottoming': [r'bottoming', r'hitting\s+the\s+track'],
    }

    for issue_key, patterns in issue_patterns.items():
        for pattern in patterns:
            # Find mentions of this issue
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context (100 chars before/after)
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 200)
                context = text[start:end]

                if issue_key not in guides:
                    guides[issue_key] = []
                guides[issue_key].append(context.strip())

    return guides


def _extract_parameter_interactions(text: str) -> Dict:
    """
    Extract information about parameter interactions.

    Looks for mentions like:
    - "Increasing spring rate will also affect..."
    - "Tire pressure and spring rate work together to..."
    """
    interactions = {}

    # This is complex to extract automatically, so we'll populate
    # with known NASCAR interactions for now
    interactions = {
        'tire_psi_rr': {
            'affects': ['rear_grip', 'tire_temp', 'spring_rate_effect'],
            'interacts_with': ['spring_rr', 'track_bar_height'],
            'note': 'Primary rear grip adjustment. Affects how springs work.'
        },
        'cross_weight': {
            'affects': ['left_rear_load', 'forward_bite', 'corner_entry'],
            'interacts_with': ['track_bar_height', 'spring_rates'],
            'note': 'Changes weight distribution. Affects tire loading.'
        },
    }

    return interactions


def _load_default_nascar_knowledge() -> Dict:
    """
    Load default NASCAR setup knowledge (fallback when no PDF available).

    Based on common NASCAR setup principles.
    """
    return {
        'parameter_limits': {
            'tire_psi_lf': {'min': 20.0, 'max': 35.0, 'unit': 'psi', 'typical': '26-30'},
            'tire_psi_rf': {'min': 20.0, 'max': 40.0, 'unit': 'psi', 'typical': '32-38'},
            'tire_psi_lr': {'min': 18.0, 'max': 35.0, 'unit': 'psi', 'typical': '24-28'},
            'tire_psi_rr': {'min': 18.0, 'max': 40.0, 'unit': 'psi', 'typical': '28-34'},
            'cross_weight': {'min': 48.0, 'max': 56.0, 'unit': '%', 'typical': '50-54'},
            'track_bar_height_left': {'min': -50.0, 'max': 50.0, 'unit': 'mm', 'typical': '-10 to +20'},
            'spring_lf': {'min': 200, 'max': 700, 'unit': 'N/mm', 'typical': '350-500'},
            'spring_rf': {'min': 200, 'max': 700, 'unit': 'N/mm', 'typical': '400-600'},
            'spring_lr': {'min': 150, 'max': 450, 'unit': 'N/mm', 'typical': '200-350'},
            'spring_rr': {'min': 150, 'max': 450, 'unit': 'N/mm', 'typical': '250-400'},
        },
        'handling_guides': {
            'loose_exit': {
                'description': 'Car is loose (oversteer) on corner exit',
                'typical_causes': ['Low rear tire pressure', 'Soft rear springs', 'Too much cross weight'],
                'recommended_changes': [
                    {'param': 'tire_psi_rr', 'direction': 'increase', 'priority': 1, 'notes': 'Primary adjustment for exit oversteer'},
                    {'param': 'tire_psi_lr', 'direction': 'increase', 'priority': 2, 'notes': 'Secondary rear grip adjustment'},
                    {'param': 'spring_rr', 'direction': 'increase', 'priority': 3, 'notes': 'If tire pressure at limit'},
                    {'param': 'cross_weight', 'direction': 'decrease', 'priority': 4, 'notes': 'Reduces left rear loading'},
                ],
                'caution': 'Rear tire pressure is most effective but has narrow window'
            },
            'loose_entry': {
                'description': 'Car is loose (oversteer) on corner entry',
                'typical_causes': ['Low rear grip at initial turn-in', 'High cross weight', 'Front tire pressure too high'],
                'recommended_changes': [
                    {'param': 'tire_psi_lf', 'direction': 'decrease', 'priority': 1, 'notes': 'Increase front grip'},
                    {'param': 'tire_psi_rr', 'direction': 'increase', 'priority': 2, 'notes': 'Stabilize rear'},
                    {'param': 'cross_weight', 'direction': 'decrease', 'priority': 3, 'notes': 'Reduce rear bias'},
                ],
                'caution': 'Entry oversteer less common than exit oversteer'
            },
            'tight_understeer': {
                'description': 'Car is tight (understeer) or pushing',
                'typical_causes': ['Low front tire pressure', 'Soft front springs', 'Too little cross weight'],
                'recommended_changes': [
                    {'param': 'tire_psi_lf', 'direction': 'increase', 'priority': 1, 'notes': 'Primary front grip adjustment'},
                    {'param': 'tire_psi_rf', 'direction': 'increase', 'priority': 2, 'notes': 'Right front carries most load'},
                    {'param': 'cross_weight', 'direction': 'increase', 'priority': 3, 'notes': 'Adds front grip'},
                    {'param': 'spring_lf', 'direction': 'increase', 'priority': 4, 'notes': 'If tire pressure at limit'},
                ],
                'caution': 'Too much front tire pressure can make car feel harsh'
            },
            'bottoming': {
                'description': 'Car is bottoming out or hitting track surface',
                'typical_causes': ['Springs too soft', 'Ride height too low', 'Heavy braking/cornering loads'],
                'recommended_changes': [
                    {'param': 'spring_lf', 'direction': 'increase', 'priority': 1, 'notes': 'Stiffen front if bottoming in corners'},
                    {'param': 'spring_rf', 'direction': 'increase', 'priority': 1, 'notes': 'Stiffen front if bottoming in corners'},
                    {'param': 'spring_lr', 'direction': 'increase', 'priority': 2, 'notes': 'Stiffen rear if bottoming on exit'},
                    {'param': 'spring_rr', 'direction': 'increase', 'priority': 2, 'notes': 'Stiffen rear if bottoming on exit'},
                ],
                'caution': 'Stiffer springs reduce mechanical grip'
            },
        },
        'parameter_interactions': {
            'tire_psi_rr': {
                'affects': ['rear_grip', 'tire_temp', 'tire_wear', 'corner_exit_traction'],
                'interacts_with': ['spring_rr', 'track_bar_height', 'cross_weight'],
                'notes': 'Most sensitive parameter. Small changes (0.5 psi) have large effects. Too low = loose exit, too high = tight and harsh.'
            },
            'tire_psi_lr': {
                'affects': ['rear_grip', 'tire_temp', 'left_rear_traction'],
                'interacts_with': ['spring_lr', 'cross_weight'],
                'notes': 'Less impactful than RR but still important. Typically run 2-4 psi lower than RR on ovals.'
            },
            'cross_weight': {
                'affects': ['weight_distribution', 'left_rear_load', 'forward_bite', 'corner_entry_balance'],
                'interacts_with': ['track_bar_height', 'spring_rates', 'tire_pressures'],
                'notes': 'Changes load on all tires. More CW = tighter entry, looser exit. Typically 50-54% on short tracks.'
            },
            'spring_rates': {
                'affects': ['ride_height', 'mechanical_grip', 'platform_control', 'aero_balance'],
                'interacts_with': ['tire_pressures', 'shocks', 'sway_bars'],
                'notes': 'Higher rates = less grip but more platform control. Interact strongly with tire pressures.'
            },
        },
        'best_practices': {
            'testing_approach': [
                'Change one parameter at a time (or front/rear pair)',
                'Make small incremental changes (0.5 psi, 25 N/mm springs)',
                'Run at least 3-5 laps after each change',
                'Always return to baseline if change makes car worse',
                'Track temperature and track rubber conditions',
            ],
            'parameter_change_sizes': {
                'tire_psi': '0.5-1.0 psi increments',
                'cross_weight': '0.5% increments (equivalent to ~1-2 turns on jack bolt)',
                'springs': '25-50 N/mm increments',
                'track_bar': '5-10 mm increments',
            },
        },
    }


def get_relevant_knowledge(knowledge_base: Dict, handling_issue: str = None, parameter: str = None) -> Dict:
    """
    Get relevant sections of knowledge base for current situation.

    Args:
        knowledge_base: Output from load_setup_manual()
        handling_issue: Current handling complaint (e.g., 'loose_exit')
        parameter: Specific parameter to get info about

    Returns:
        Dict with relevant limits, guides, and interactions
    """
    relevant = {
        'limits': {},
        'guidance': {},
        'interactions': {},
        'warnings': []
    }

    # Get parameter limits if specified
    if parameter and parameter in knowledge_base.get('parameter_limits', {}):
        relevant['limits'][parameter] = knowledge_base['parameter_limits'][parameter]

    # Get handling guidance if specified
    if handling_issue and handling_issue in knowledge_base.get('handling_guides', {}):
        relevant['guidance'] = knowledge_base['handling_guides'][handling_issue]

    # Get parameter interactions if specified
    if parameter and parameter in knowledge_base.get('parameter_interactions', {}):
        relevant['interactions'] = knowledge_base['parameter_interactions'][parameter]

    return relevant


def format_knowledge_for_llm(knowledge: Dict) -> str:
    """
    Format knowledge base sections for LLM context.

    Args:
        knowledge: Output from get_relevant_knowledge()

    Returns:
        Formatted string suitable for LLM prompt
    """
    sections = []

    if knowledge.get('limits'):
        sections.append("**Parameter Limits:**")
        for param, limits in knowledge['limits'].items():
            typical = limits.get('typical', '')
            sections.append(f"  - {param}: {limits['min']}-{limits['max']} {limits['unit']} (typical: {typical})")

    if knowledge.get('guidance'):
        guidance = knowledge['guidance']
        sections.append("\n**Setup Guidance:**")
        if 'description' in guidance:
            sections.append(f"  {guidance['description']}")
        if 'recommended_changes' in guidance:
            sections.append("  Recommended changes:")
            for change in guidance['recommended_changes'][:3]:  # Top 3
                sections.append(f"    {change['priority']}. {change['param']} - {change['direction']} ({change['notes']})")
        if 'caution' in guidance:
            sections.append(f"  ⚠️ {guidance['caution']}")

    if knowledge.get('interactions'):
        interactions = knowledge['interactions']
        sections.append("\n**Parameter Interactions:**")
        if 'affects' in interactions:
            sections.append(f"  Affects: {', '.join(interactions['affects'])}")
        if 'interacts_with' in interactions:
            sections.append(f"  Interacts with: {', '.join(interactions['interacts_with'])}")
        if 'notes' in interactions:
            sections.append(f"  Notes: {interactions['notes']}")

    return "\n".join(sections) if sections else "No specific guidance available."


if __name__ == "__main__":
    # Test the knowledge base
    print("Testing Knowledge Base Loader:\n")

    kb = load_setup_manual()
    print(f"\nKnowledge base source: {kb['source']}")
    print(f"Parameter limits loaded: {len(kb['parameter_limits'])}")
    print(f"Handling guides loaded: {len(kb['handling_guides'])}")

    # Test getting relevant knowledge
    print("\n--- Example: Loose on exit ---")
    relevant = get_relevant_knowledge(kb, handling_issue='loose_exit', parameter='tire_psi_rr')
    formatted = format_knowledge_for_llm(relevant)
    print(formatted)
