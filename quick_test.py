"""Quick test to verify basic functionality"""
import sys
print("Starting test...")
sys.stdout.flush()

print("Step 1: Testing imports...")
sys.stdout.flush()

from pathlib import Path
import pandas as pd
import numpy as np

print("Step 2: Testing parser imports...")
sys.stdout.flush()

from ibt_parser import IBTParser
from telemetry_parser import TelemetryParser

print("Step 3: Testing race engineer import...")
sys.stdout.flush()

from race_engineer import create_race_engineer_workflow

print("Step 4: Creating IBT parser...")
sys.stdout.flush()

parser = IBTParser()
print(f"Parser created. Has library: {parser.has_ibt_library}")
sys.stdout.flush()

print("Step 5: Generating mock data...")
sys.stdout.flush()

mock_data = parser._generate_mock_telemetry(Path("test.ibt"))
print(f"Mock data generated: {len(mock_data)} rows")
sys.stdout.flush()

print("Step 6: Testing workflow creation...")
sys.stdout.flush()

app = create_race_engineer_workflow()
print("Workflow created successfully")
sys.stdout.flush()

print("\n[SUCCESS] All tests passed!")
