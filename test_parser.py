"""
Quick Start: Test Telemetry Parser
Run this to verify everything works with your sample file
"""

from pathlib import Path
import sys
import json

# Add current directory to path for imports
sys.path.append('.')

def test_telemetry_parsing():
    """Test the telemetry parser with Bristol data"""

    print(" Bristol AI Race Engineer - Telemetry Parser Test")
    print("=" * 60)

    # Import our parser
    from telemetry_parser import TelemetryParser

    # Look for any .ldx files in bristol_data directory
    ldx_files = list(Path("bristol_data").glob("*.ldx"))

    if not ldx_files:
        print("[ERROR] No .ldx files found in bristol_data/ directory")
        print("[INFO]  This test requires actual telemetry files")
        print("[INFO]  For full demo with mock data, run: python main.py")
        return None, None

    # Use the first file found
    sample_file = ldx_files[0]

    print(f"\n📁 Testing with: {sample_file.name}")
    print("-" * 40)

    # Parse the file
    parser = TelemetryParser()
    data = parser.parse_ldx_file(sample_file)
    
    # Display key setup parameters
    print("\n[TOOL] Setup Parameters Extracted:")
    print(f"   Cross Weight: {data.get('cross_weight', 'N/A')}%")
    print(f"   Nose Weight: {data.get('nose_weight', 'N/A')}%")
    
    print(f"\n[DATA] Tire Pressures (PSI):")
    print(f"   Left Front:  {data.get('tire_psi_lf', 0):.1f}")
    print(f"   Right Front: {data.get('tire_psi_rf', 0):.1f}")
    print(f"   Left Rear:   {data.get('tire_psi_lr', 0):.1f}")
    print(f"   Right Rear:  {data.get('tire_psi_rr', 0):.1f}")
    
    print(f"\n🏎️ Spring Rates (N/mm):")
    print(f"   Left Front:  {data.get('spring_lf', 'N/A')}")
    print(f"   Right Front: {data.get('spring_rf', 'N/A')}")
    print(f"   Left Rear:   {data.get('spring_lr', 'N/A')}")
    print(f"   Right Rear:  {data.get('spring_rr', 'N/A')}")
    
    print(f"\n📏 Ride Heights (mm):")
    print(f"   Left Front:  {data.get('ride_height_lf', 'N/A')}")
    print(f"   Right Front: {data.get('ride_height_rf', 'N/A')}")
    print(f"   Left Rear:   {data.get('ride_height_lr', 'N/A')}")
    print(f"   Right Rear:  {data.get('ride_height_rr', 'N/A')}")
    
    print(f"\n⏱️ Performance Data:")
    print(f"   Track: {data.get('venue', 'N/A')}")
    print(f"   Total Laps: {data.get('total_laps', 'N/A')}")
    print(f"   Fastest Lap: {data.get('fastest_time', 'N/A')}")
    print(f"   Track Temp: {data.get('track_temp', 'N/A')}°C")
    
    # Save to JSON for inspection
    output_file = Path("parsed_telemetry_sample.json")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"\n[OK] Success! Full data saved to: {output_file}")
    print("\n" + "=" * 60)
    
    # Now test the IBT parser with mock data
    print("\n🎮 Testing IBT Parser (Mock Telemetry Generation)")
    print("-" * 40)
    
    from ibt_parser import IBTParser
    
    ibt_parser = IBTParser()
    mock_telemetry = ibt_parser._generate_mock_telemetry(Path("bristol_stint_1.ibt"))
    
    print(f"Generated {len(mock_telemetry)} telemetry samples")
    print(f"Channels available: {', '.join(mock_telemetry.columns[:5])}...")
    
    # Extract lap statistics
    lap_stats = ibt_parser.extract_lap_statistics(mock_telemetry)
    
    print(f"\n[DATA] Lap Statistics:")
    print(f"   Laps analyzed: {len(lap_stats)}")
    print(f"   Best lap time: {lap_stats['lap_time'].min():.3f} seconds")
    print(f"   Average lap time: {lap_stats['lap_time'].mean():.3f} seconds")
    print(f"   Consistency (StdDev): {lap_stats['lap_time'].std():.3f} seconds")
    
    # Save lap stats
    lap_stats_file = Path("lap_statistics_sample.csv")
    lap_stats.to_csv(lap_stats_file, index=False)
    print(f"\n[OK] Lap statistics saved to: {lap_stats_file}")
    
    print("\n" + "=" * 60)
    print(" Telemetry pipeline verified and ready!")
    print("\nNext steps:")
    print("1. Collect your Bristol test data (30 runs)")
    print("2. Export as .ldx files from MoTec")
    print("3. Run parsers on full dataset")
    print("4. Feed to AI agents for analysis")
    
    return data, lap_stats

if __name__ == "__main__":
    data, lap_stats = test_telemetry_parsing()
