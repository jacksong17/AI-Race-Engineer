"""
Test the recommendation parsing fix
"""
from race_engineer.agents import _parse_recommendations_from_text


def test_recommendation_parsing():
    """Test that the parsing works for various formats"""

    # Mock state with empty statistical analysis
    mock_state = {
        'statistical_analysis': None
    }

    # Test 1: Multi-word parameter name (the original issue)
    text1 = "Recommend: Increase rear spring rate by 50 lb/in"
    recs1 = _parse_recommendations_from_text(text1, mock_state)
    print(f"Test 1 - Multi-word parameter:")
    print(f"  Input: {text1}")
    print(f"  Parsed: {recs1}")
    assert len(recs1) == 1, "Should parse 1 recommendation"
    assert recs1[0]['parameter'] == 'spring_rr', f"Should map to spring_rr, got {recs1[0]['parameter']}"
    assert recs1[0]['direction'] == 'increase'
    assert recs1[0]['magnitude'] == 50
    assert recs1[0]['magnitude_unit'] == 'lb/in'
    print("  ✓ PASSED\n")

    # Test 2: Underscore parameter name
    text2 = "Recommend: decrease tire_psi_rr by 1.5 PSI"
    recs2 = _parse_recommendations_from_text(text2, mock_state)
    print(f"Test 2 - Underscore parameter:")
    print(f"  Input: {text2}")
    print(f"  Parsed: {recs2}")
    assert len(recs2) == 1
    assert recs2[0]['parameter'] == 'tire_psi_rr'
    assert recs2[0]['direction'] == 'decrease'
    assert recs2[0]['magnitude'] == 1.5
    assert recs2[0]['magnitude_unit'] == 'PSI'
    print("  ✓ PASSED\n")

    # Test 3: Multiple recommendations
    text3 = """
    Recommend: decrease tire_psi_rr by 1.5 PSI
    Recommend: increase cross_weight by 0.5 %
    """
    recs3 = _parse_recommendations_from_text(text3, mock_state)
    print(f"Test 3 - Multiple recommendations:")
    print(f"  Input: {text3}")
    print(f"  Parsed: {recs3}")
    assert len(recs3) == 2
    assert recs3[0]['parameter'] == 'tire_psi_rr'
    assert recs3[1]['parameter'] == 'cross_weight'
    print("  ✓ PASSED\n")

    # Test 4: Mapped parameter name
    text4 = "Recommend: decrease RR tire pressure by 2 PSI"
    recs4 = _parse_recommendations_from_text(text4, mock_state)
    print(f"Test 4 - Mapped parameter name:")
    print(f"  Input: {text4}")
    print(f"  Parsed: {recs4}")
    assert len(recs4) == 1
    assert recs4[0]['parameter'] == 'tire_psi_rr', f"Should map to tire_psi_rr, got {recs4[0]['parameter']}"
    print("  ✓ PASSED\n")

    # Test 5: Different action verbs
    text5 = """
    Recommend: reduce tire_psi_lf by 1 PSI
    Recommend: raise cross_weight by 0.5 %
    """
    recs5 = _parse_recommendations_from_text(text5, mock_state)
    print(f"Test 5 - Different action verbs:")
    print(f"  Input: {text5}")
    print(f"  Parsed: {recs5}")
    assert len(recs5) == 2
    assert recs5[0]['direction'] == 'decrease'  # reduce -> decrease
    assert recs5[1]['direction'] == 'increase'  # raise -> increase
    print("  ✓ PASSED\n")

    print("="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)


if __name__ == '__main__':
    test_recommendation_parsing()
