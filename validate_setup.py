"""
Validation Script - Test all components before running full demo
Run this first to ensure all dependencies and code are working
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("[*] Testing imports...")

    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib.pyplot'),
        ('seaborn', 'seaborn'),
        ('sklearn', 'scikit-learn'),
        ('langgraph', 'langgraph'),
        ('lxml', 'lxml'),
    ]

    optional_packages = [
        ('irsdk', 'pyirsdk'),
    ]

    all_good = True

    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"   [OK] {package_name}")
        except ImportError:
            print(f"   [FAIL] {package_name} - REQUIRED")
            all_good = False

    for module_name, package_name in optional_packages:
        try:
            __import__(module_name)
            print(f"   [OK] {package_name}")
        except ImportError:
            print(f"   [WARN] {package_name} - OPTIONAL (will use mock data)")

    return all_good

def test_project_structure():
    """Verify directory structure exists"""
    print("\n[*] Testing project structure...")

    required_dirs = [
        'bristol_data',
        'data/raw/telemetry',
        'data/processed',
        'output'
    ]

    all_good = True

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"   [OK] {dir_path}/")
        else:
            print(f"   [FAIL] {dir_path}/ - MISSING")
            all_good = False

    return all_good

def test_code_files():
    """Verify all Python modules are present and can be imported"""
    print("\n[*] Testing Python modules...")

    required_files = [
        'telemetry_parser.py',
        'ibt_parser.py',
        'race_engineer.py',
        'create_visualizations.py',
        'main.py'
    ]

    all_good = True

    for file_name in required_files:
        path = Path(file_name)
        if not path.exists():
            print(f"   [FAIL] {file_name} - MISSING")
            all_good = False
            continue

        # Try to import it
        try:
            module_name = file_name.replace('.py', '')
            __import__(module_name)
            print(f"   [OK] {file_name}")
        except Exception as e:
            print(f"   [WARN] {file_name} - Import error: {e}")
            all_good = False

    return all_good

def test_telemetry_parser():
    """Test the telemetry parser"""
    print("\n[*] Testing TelemetryParser...")

    try:
        from telemetry_parser import TelemetryParser

        parser = TelemetryParser()
        print("   [OK] TelemetryParser initialized")
        return True
    except Exception as e:
        print(f"   [FAIL] TelemetryParser error: {e}")
        return False

def test_ibt_parser():
    """Test the IBT parser"""
    print("\n[*] Testing IBTParser...")

    try:
        from ibt_parser import IBTParser

        parser = IBTParser()
        print(f"   [OK] IBTParser initialized")

        if parser.has_ibt_library:
            print("   [OK] pyirsdk available - can parse real .ibt files")
        else:
            print("   [WARN] pyirsdk not available - will use mock data")

        # Test mock data generation
        import pandas as pd
        mock_data = parser._generate_mock_telemetry(Path("test.ibt"))
        if isinstance(mock_data, pd.DataFrame) and not mock_data.empty:
            print(f"   [OK] Mock telemetry generation works ({len(mock_data)} samples)")
        else:
            print("   [FAIL] Mock telemetry generation failed")
            return False

        return True
    except Exception as e:
        print(f"   [FAIL] IBTParser error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_race_engineer():
    """Test the race engineer workflow"""
    print("\n[*] Testing Race Engineer Workflow...")

    try:
        from race_engineer import create_race_engineer_workflow, RaceEngineerState

        app = create_race_engineer_workflow()
        print("   [OK] Workflow created successfully")
        print("   [OK] Graph compiled")

        return True
    except Exception as e:
        print(f"   [FAIL] Race Engineer error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_files():
    """Check for telemetry data files"""
    print("\n[*] Checking for data files...")

    ibt_files = list(Path("data/raw/telemetry").glob("*.ibt"))
    ldx_files = list(Path("bristol_data").glob("*.ldx"))

    print(f"   [INFO] .ibt files: {len(ibt_files)}")
    print(f"   [INFO] .ldx files: {len(ldx_files)}")

    if ibt_files:
        for f in ibt_files:
            print(f"      - {f.name}")

    if ldx_files:
        for f in ldx_files:
            print(f"      - {f.name}")

    if not ibt_files and not ldx_files:
        print("   [INFO] No telemetry files found - demo will use mock data")

    return True

def main():
    """Run all validation tests"""
    print("="*70)
    print("  BRISTOL AI RACE ENGINEER - SETUP VALIDATION")
    print("="*70)

    results = []

    results.append(("Package Imports", test_imports()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("Code Files", test_code_files()))
    results.append(("TelemetryParser", test_telemetry_parser()))
    results.append(("IBTParser", test_ibt_parser()))
    results.append(("Race Engineer", test_race_engineer()))
    results.append(("Data Files", test_data_files()))

    print("\n" + "="*70)
    print("  VALIDATION SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {status:10s} - {test_name}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("  [SUCCESS] ALL TESTS PASSED!")
        print("  Ready to run: python main.py")
    else:
        print("  [ERROR] SOME TESTS FAILED")
        print("  Please fix errors before running main.py")
        print()
        print("  Common fixes:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Ensure all .py files are present")
        print("  3. Check that directory structure was created")
        return 1

    print("="*70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
