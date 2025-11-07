"""
iRacing Telemetry Parser for AI Race Engineer
Extracts setup parameters and performance metrics from .ldx files
"""

import xml.etree.ElementTree as ET
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

class TelemetryParser:
    """Parse iRacing telemetry files (.ldx format from MoTec export)"""
    
    # Key setup parameters we want to track for Bristol Truck optimization
    SETUP_PARAMS = {
        # Tire Pressures (converted from kPa to PSI for familiarity)
        'tire_psi_lf': 'CarSetup_Tires_LeftFront_ColdPressure',
        'tire_psi_rf': 'CarSetup_Tires_RightFront_ColdPressure',
        'tire_psi_lr': 'CarSetup_Tires_LeftRear_ColdPressure',
        'tire_psi_rr': 'CarSetup_Tires_RightRear_ColdPressure',
        
        # Chassis Setup
        'cross_weight': 'CarSetup_Chassis_Front_CrossWeight',
        'nose_weight': 'CarSetup_Chassis_Front_NoseWeight',
        'track_bar_height_left': 'CarSetup_Chassis_LeftRear_TrackBarHeight',
        'track_bar_height_right': 'CarSetup_Chassis_RightRear_TrackBarHeight',
        
        # Springs (N/mm)
        'spring_lf': 'CarSetup_Chassis_LeftFront_SpringRate',
        'spring_rf': 'CarSetup_Chassis_RightFront_SpringRate',
        'spring_lr': 'CarSetup_Chassis_LeftRear_SpringRate',
        'spring_rr': 'CarSetup_Chassis_RightRear_SpringRate',
        
        # Ride Heights
        'ride_height_lf': 'CarSetup_Chassis_LeftFront_RideHeight',
        'ride_height_rf': 'CarSetup_Chassis_RightFront_RideHeight',
        'ride_height_lr': 'CarSetup_Chassis_LeftRear_RideHeight',
        'ride_height_rr': 'CarSetup_Chassis_RightRear_RideHeight',
        
        # Alignment
        'camber_lf': 'CarSetup_Chassis_LeftFront_Camber',
        'camber_rf': 'CarSetup_Chassis_RightFront_Camber',
        'toe_lf': 'CarSetup_Chassis_LeftFront_ToeIn',
        'toe_rf': 'CarSetup_Chassis_RightFront_ToeIn',
        
        # Driver Controls
        'steering_ratio': 'CarSetup_Chassis_Front_SteeringRatio',
        'brake_bias': 'CarSetup_Chassis_Front_FrontBrakeBias',
        
        # ARB
        'arb_diameter': 'CarSetup_Chassis_FrontArb_Diameter',
        'arb_attach': 'CarSetup_Chassis_FrontArb_Attach',
    }
    
    # Performance metrics
    PERF_METRICS = {
        'venue': 'Venue',
        'track_temp': 'Track Temp',
        'air_temp': 'Air Temp',
        'total_laps': 'Total Laps',
        'fastest_time': 'Fastest Time',
        'fastest_lap_num': 'Fastest Lap'
    }
    
    def __init__(self):
        self.data = []
        
    def parse_ldx_file(self, filepath: Path) -> Dict:
        """Parse a single .ldx file and extract setup and performance data"""
        
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Extract session info
        session_data = {
            'filename': filepath.stem,
            'session_id': self._extract_session_id(filepath.stem)
        }
        
        # Parse all numeric and string values from Details section
        details = {}
        for element in root.find('.//Details'):
            if element.tag == 'Numeric':
                key = element.get('Id')
                value = float(element.get('Value', 0))
                unit = element.get('Unit', '')
                
                # Convert kPa to PSI for tire pressures
                if unit == 'kPa' and 'Pressure' in key:
                    value = value * 0.145038  # kPa to PSI conversion
                    
                details[key] = value
                
            elif element.tag == 'String':
                key = element.get('Id')
                value = element.get('Value', '')
                details[key] = value
        
        # Extract setup parameters
        for param_name, xml_key in self.SETUP_PARAMS.items():
            if xml_key in details:
                session_data[param_name] = details[xml_key]
            else:
                session_data[param_name] = None
                
        # Extract performance metrics
        for metric_name, xml_key in self.PERF_METRICS.items():
            if xml_key in details:
                value = details[xml_key]
                # Convert lap time string to seconds
                if metric_name == 'fastest_time' and isinstance(value, str):
                    session_data[metric_name] = self._parse_lap_time(value)
                else:
                    session_data[metric_name] = value
            else:
                session_data[metric_name] = None
                
        return session_data
    
    def _extract_session_id(self, filename: str) -> str:
        """Extract session identifier from filename"""
        # Example: trucks_silverado2019_phoenix_2021_ovalopen_2025-11-01_19-24-11_Stint_1
        parts = filename.split('_')
        if 'Stint' in filename:
            return f"{parts[-2]}_{parts[-1]}"  # Stint_1, Stint_2, etc.
        return filename[-10:]  # Last 10 chars as fallback
    
    def _parse_lap_time(self, time_str: str) -> float:
        """Convert lap time string (M:SS.mmm) to seconds"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            return float(time_str)
        except:
            return None
    
    def process_directory(self, directory: Path) -> pd.DataFrame:
        """Process all .ldx files in a directory"""
        ldx_files = list(directory.glob("*.ldx"))
        
        print(f"Found {len(ldx_files)} telemetry files")
        
        for filepath in ldx_files:
            print(f"Processing: {filepath.name}")
            session_data = self.parse_ldx_file(filepath)
            self.data.append(session_data)
            
        df = pd.DataFrame(self.data)
        
        # Calculate derived features
        df = self._add_derived_features(df)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated features useful for analysis"""
        
        # Tire pressure differentials
        df['tire_psi_stagger_front'] = df['tire_psi_rf'] - df['tire_psi_lf']
        df['tire_psi_stagger_rear'] = df['tire_psi_rr'] - df['tire_psi_lr']
        df['tire_psi_split'] = ((df['tire_psi_lf'] + df['tire_psi_rf']) / 2) - \
                                ((df['tire_psi_lr'] + df['tire_psi_rr']) / 2)
        
        # Spring rate ratios
        df['spring_ratio_front'] = df['spring_rf'] / df['spring_lf'] if 'spring_lf' in df else None
        df['spring_ratio_diagonal'] = (df['spring_rf'] + df['spring_lr']) / \
                                      (df['spring_lf'] + df['spring_rr']) if 'spring_lf' in df else None
        
        # Ride height rake
        df['rake'] = ((df['ride_height_lf'] + df['ride_height_rf']) / 2) - \
                     ((df['ride_height_lr'] + df['ride_height_rr']) / 2)
        
        # Track bar split
        df['track_bar_split'] = df['track_bar_height_left'] - df['track_bar_height_right']
        
        return df
    
    def export_for_agents(self, df: pd.DataFrame, output_path: Path):
        """Export processed data in format ready for AI agents"""
        
        # Save as CSV for agents
        csv_path = output_path / 'telemetry_processed.csv'
        df.to_csv(csv_path, index=False)
        print(f"Exported to {csv_path}")
        
        # Save summary statistics for agent context
        summary = {
            'total_sessions': len(df),
            'tracks': df['venue'].unique().tolist() if 'venue' in df else [],
            'fastest_lap': df['fastest_time'].min() if 'fastest_time' in df else None,
            'avg_lap': df['fastest_time'].mean() if 'fastest_time' in df else None,
            'parameter_ranges': {}
        }
        
        # Calculate parameter ranges for agent understanding
        for param in self.SETUP_PARAMS.keys():
            if param in df and df[param].notna().any():
                summary['parameter_ranges'][param] = {
                    'min': float(df[param].min()),
                    'max': float(df[param].max()),
                    'mean': float(df[param].mean()),
                    'std': float(df[param].std())
                }
        
        summary_path = output_path / 'telemetry_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary exported to {summary_path}")
        
        return csv_path, summary_path


# Demo usage
if __name__ == "__main__":
    parser = TelemetryParser()
    
    # Example: Process single file
    sample_file = Path("trucks_silverado2019_phoenix_2021_ovalopen_2025-11-01_19-24-11_Stint_1.ldx")
    if sample_file.exists():
        data = parser.parse_ldx_file(sample_file)
        print("\nExtracted Setup Data:")
        print(json.dumps(data, indent=2))
    
    # Example: Process directory of telemetry files
    # telemetry_dir = Path("./telemetry_files")
    # df = parser.process_directory(telemetry_dir)
    # parser.export_for_agents(df, Path("./data/processed"))
