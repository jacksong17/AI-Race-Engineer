"""
Native iRacing IBT Telemetry Parser
Direct parsing of .ibt files without MoTec conversion
This demonstrates advanced data pipeline capabilities
"""

import struct
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# --- THIS IS THE CRITICAL FIX ---
# We are importing 'irsdk', which is provided by the 'pyirsdk' package
# We are no longer trying to import 'python-ibt', which was incorrect
try:
    import irsdk
    PYIRSDK_INSTALLED = True
except ImportError:
    PYIRSDK_INSTALLED = False
# ---------------------------------

class IBTParser:
    """
    Parse native iRacing .ibt telemetry files
    """
    
    # Bristol-specific channels we want to extract
    CHANNELS_OF_INTEREST = [
        'Speed', 'Throttle', 'Brake', 'SteeringWheelAngle',
        'LapCurrentLapTime', 'LapDist', 'LapDistPct',
        'LFtempCL', 'LFtempCM', 'LFtempCR', 'RFtempCL', 'RFtempCM', 'RFtempCR',
        'LRtempCL', 'LRtempCM', 'LRtempCR', 'RRtempCL', 'RRtempCM', 'RRtempCR',
        'LFshockDefl', 'RFshockDefl', 'LRshockDefl', 'RRshockDefl',
        'LongAccel', 'LatAccel', 'VertAccel',
        'RPM', 'Gear', 'SessionTime', 'SessionNum', 'SessionLapsRemainEx'
    ]
    
    def __init__(self):
        """Initialize the IBT parser"""
        self.has_ibt_library = PYIRSDK_INSTALLED
        if not self.has_ibt_library:
            print("="*60)
            print("⚠️ WARNING: 'pyirsdk' library not found. Using MOCK data for demo.")
            print("   To parse real .ibt files, run: pip install pyirsdk")
            print("="*60)
    
    def parse_ibt_file(self, filepath: Path) -> pd.DataFrame:
        """
        Parse an .ibt telemetry file
        Returns DataFrame with telemetry channels
        """
        
        if self.has_ibt_library:
            return self._parse_ibt_native(filepath)
        else:
            return self._generate_mock_telemetry(filepath)
    
    def _parse_ibt_native(self, filepath: Path) -> pd.DataFrame:
        """Parse using pyirsdk library (if available)"""
        filepath = Path(filepath) if not isinstance(filepath, Path) else filepath
        print(f"   > Parsing real .ibt file: {filepath.name}")
        try:
            # Open the .ibt file
            ir = irsdk.IBT()
            ir.open(str(filepath))
            
            # Extract data for each channel
            data = {}
            for channel in self.CHANNELS_OF_INTEREST:
                if channel in ir:
                    # Read all samples for this channel at once
                    data[channel] = ir[channel].data
                else:
                    print(f"   > Warning: Channel '{channel}' not found in {filepath.name}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            ir.close()
            
            return df
            
        except Exception as e:
            print(f"❌ ERROR: Failed to parse .ibt file: {e}")
            print("   > Falling back to mock telemetry.")
            return self._generate_mock_telemetry(filepath)
    
    def _generate_mock_telemetry(self, filepath: Path) -> pd.DataFrame:
        """
        Generate realistic mock telemetry data for demo purposes
        This simulates what real Bristol truck telemetry would look like
        """
        filepath = Path(filepath) if not isinstance(filepath, Path) else filepath
        print(f"   > Generating MOCK telemetry for demo (simulating {filepath.name})")
        
        # (Rest of the mock data generation code is identical)
        samples_per_lap = 930
        num_laps = 10
        total_samples = samples_per_lap * num_laps
        
        time = np.linspace(0, 15.5 * num_laps, total_samples)
        lap_dist_pct = np.tile(np.linspace(0, 1, samples_per_lap), num_laps)
        
        data = {
            'SessionTime': time,
            'LapDistPct': lap_dist_pct,
            'Speed': self._generate_speed_trace(lap_dist_pct, samples_per_lap),
            'Throttle': self._generate_throttle_trace(lap_dist_pct),
            'Brake': self._generate_brake_trace(lap_dist_pct),
            'SteeringWheelAngle': self._generate_steering_trace(lap_dist_pct),
            'RPM': self._generate_rpm_trace(lap_dist_pct),
            'Gear': np.where(lap_dist_pct < 0.1, 3, 4),
            'LatAccel': self._generate_lat_accel(lap_dist_pct),
            'LongAccel': self._generate_long_accel(lap_dist_pct),
            'LFtempCM': 180 + np.arange(total_samples) * 0.001 + np.random.normal(0, 2, total_samples),
            'RFtempCM': 195 + np.arange(total_samples) * 0.0015 + np.random.normal(0, 2, total_samples),
            'LRtempCM': 175 + np.arange(total_samples) * 0.0008 + np.random.normal(0, 2, total_samples),
            'RRtempCM': 190 + np.arange(total_samples) * 0.0012 + np.random.normal(0, 2, total_samples),
        }
        
        lap_times = 15.5 + np.random.normal(0, 0.15, num_laps)
        lap_current = np.repeat(np.arange(num_laps), samples_per_lap)
        lap_time_current = np.zeros(total_samples)
        
        for lap in range(num_laps):
            lap_mask = lap_current == lap
            lap_samples = np.sum(lap_mask)
            lap_time_current[lap_mask] = np.linspace(0, lap_times[lap], lap_samples)
        
        data['LapCurrentLapTime'] = lap_time_current
        data['Lap'] = lap_current
        
        return pd.DataFrame(data)

    # (All the _generate_... methods are identical)
    
    def _generate_speed_trace(self, lap_pct: np.ndarray, samples_per_lap: int) -> np.ndarray:
        base_speed = 110
        turn_1_2 = (lap_pct > 0.15) & (lap_pct < 0.45)
        turn_3_4 = (lap_pct > 0.65) & (lap_pct < 0.95)
        speed = np.full_like(lap_pct, base_speed)
        speed[turn_1_2] = 85 + 10 * np.sin(np.pi * (lap_pct[turn_1_2] - 0.15) / 0.3)
        speed[turn_3_4] = 85 + 10 * np.sin(np.pi * (lap_pct[turn_3_4] - 0.65) / 0.3)
        speed += np.random.normal(0, 1, len(speed))
        return speed
    
    def _generate_throttle_trace(self, lap_pct: np.ndarray) -> np.ndarray:
        throttle = np.ones_like(lap_pct) * 100
        turn_1_2 = (lap_pct > 0.20) & (lap_pct < 0.35)
        turn_3_4 = (lap_pct > 0.70) & (lap_pct < 0.85)
        throttle[turn_1_2] = 60 + 20 * np.sin(np.pi * (lap_pct[turn_1_2] - 0.20) / 0.15)
        throttle[turn_3_4] = 60 + 20 * np.sin(np.pi * (lap_pct[turn_3_4] - 0.70) / 0.15)
        return np.clip(throttle + np.random.normal(0, 2, len(throttle)), 0, 100)
    
    def _generate_brake_trace(self, lap_pct: np.ndarray) -> np.ndarray:
        brake = np.zeros_like(lap_pct)
        entry_1 = (lap_pct > 0.18) & (lap_pct < 0.22)
        entry_2 = (lap_pct > 0.68) & (lap_pct < 0.72)
        brake[entry_1] = 20 * np.sin(np.pi * (lap_pct[entry_1] - 0.18) / 0.04)
        brake[entry_2] = 20 * np.sin(np.pi * (lap_pct[entry_2] - 0.68) / 0.04)
        return np.clip(brake + np.random.normal(0, 1, len(brake)), 0, 100)
    
    def _generate_steering_trace(self, lap_pct: np.ndarray) -> np.ndarray:
        steering = np.zeros_like(lap_pct)
        turn_1_2 = (lap_pct > 0.15) & (lap_pct < 0.45)
        turn_3_4 = (lap_pct > 0.65) & (lap_pct < 0.95)
        steering[turn_1_2] = 15 * np.sin(np.pi * (lap_pct[turn_1_2] - 0.15) / 0.3)
        steering[turn_3_4] = 15 * np.sin(np.pi * (lap_pct[turn_3_4] - 0.65) / 0.3)
        return steering + np.random.normal(0, 0.5, len(steering))
    
    def _generate_rpm_trace(self, lap_pct: np.ndarray) -> np.ndarray:
        base_rpm = 7500
        turn_1_2 = (lap_pct > 0.20) & (lap_pct < 0.40)
        turn_3_4 = (lap_pct > 0.70) & (lap_pct < 0.90)
        rpm = np.full_like(lap_pct, base_rpm)
        rpm[turn_1_2] = 6800 + 400 * np.sin(np.pi * (lap_pct[turn_1_2] - 0.20) / 0.20)
        rpm[turn_3_4] = 6800 + 400 * np.sin(np.pi * (lap_pct[turn_3_4] - 0.70) / 0.20)
        return rpm + np.random.normal(0, 50, len(rpm))
    
    def _generate_lat_accel(self, lap_pct: np.ndarray) -> np.ndarray:
        lat_accel = np.zeros_like(lap_pct)
        turn_1_2 = (lap_pct > 0.15) & (lap_pct < 0.45)
        turn_3_4 = (lap_pct > 0.65) & (lap_pct < 0.95)
        lat_accel[turn_1_2] = 2.8 * np.sin(np.pi * (lap_pct[turn_1_2] - 0.15) / 0.3)
        lat_accel[turn_3_4] = 2.8 * np.sin(np.pi * (lap_pct[turn_3_4] - 0.65) / 0.3)
        return lat_accel + np.random.normal(0, 0.1, len(lat_accel))
    
    def _generate_long_accel(self, lap_pct: np.ndarray) -> np.ndarray:
        long_accel = np.zeros_like(lap_pct)
        braking_1 = (lap_pct > 0.18) & (lap_pct < 0.22)
        braking_2 = (lap_pct > 0.68) & (lap_pct < 0.72)
        accel_1 = (lap_pct > 0.40) & (lap_pct < 0.50)
        accel_2 = (lap_pct > 0.90) | (lap_pct < 0.10)
        long_accel[braking_1] = -0.5
        long_accel[braking_2] = -0.5
        long_accel[accel_1] = 0.3
        long_accel[accel_2] = 0.3
        return long_accel + np.random.normal(0, 0.05, len(long_accel))

    def extract_lap_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract lap-by-lap statistics from telemetry
        This is what we'll feed to the AI agents
        """
        
        lap_stats = []
        
        if 'Lap' not in df.columns:
            # Create lap column if not present
            df['Lap'] = (df['LapDistPct'].diff() < 0).cumsum()
        
        for lap in df['Lap'].unique():
            lap_data = df[df['Lap'] == lap]
            
            if len(lap_data) < 100:  # Skip incomplete laps
                continue
            
            # Use 'LapCurrentLapTime' to get the final time for the lap
            lap_time_val = lap_data['LapCurrentLapTime'].max() if 'LapCurrentLapTime' in lap_data.columns else None

            stats = {
                'lap_number': int(lap),
                'lap_time': lap_time_val,
                'speed_avg': lap_data['Speed'].mean(),
                'speed_min': lap_data['Speed'].min(),
                'speed_max': lap_data['Speed'].max(),
                'throttle_avg': lap_data['Throttle'].mean(),
                'brake_avg': lap_data['Brake'].mean(),
                'tire_temp_lf': lap_data['LFtempCM'].mean() if 'LFtempCM' in lap_data.columns else None,
                'tire_temp_rf': lap_data['RFtempCM'].mean() if 'RFtempCM' in lap_data.columns else None,
                'tire_temp_lr': lap_data['LRtempCM'].mean() if 'LRtempCM' in lap_data.columns else None,
                'tire_temp_rr': lap_data['RRtempCM'].mean() if 'RRtempCM' in lap_data.columns else None,
                'lat_accel_max': lap_data['LatAccel'].abs().max(),
                'steering_std': lap_data['SteeringWheelAngle'].std(),
            }
            
            lap_stats.append(stats)
        
        return pd.DataFrame(lap_stats)


class TelemetryAggregator:
    """
    Combines setup data (.ldx) with telemetry data (.ibt)
    Creates the complete dataset for AI agent analysis
    """
    
    def __init__(self):
        self.ldx_parser = None  # Will import from telemetry_parser.py
        self.ibt_parser = IBTParser()
        
    def create_training_dataset(self, 
                               ldx_files: List[Path], 
                               ibt_files: List[Path]) -> pd.DataFrame:
        """
        Create combined dataset from setup and telemetry files
        
        Args:
            ldx_files: List of .ldx files with setup data
            ibt_files: List of .ibt files with telemetry
        
        Returns:
            DataFrame ready for agent analysis
        """
        
        all_data = []
        
        # We need to match .ldx files to .ibt files
        # A simple way is to assume they have similar names or order
        
        ldx_map = {f.stem.rsplit('_', 1)[0]: f for f in ldx_files}
        ibt_map = {f.stem: f for f in ibt_files}

        for base_name, ldx_file in ldx_map.items():
            if base_name in ibt_map:
                ibt_file = ibt_map[base_name]
                
                print(f"\nProcessing session: {base_name}")
                
                # 1. Parse setup data from .ldx
                from telemetry_parser import TelemetryParser
                if self.ldx_parser is None:
                    self.ldx_parser = TelemetryParser()
                setup_data = self.ldx_parser.parse_ldx_file(ldx_file)
                
                # 2. Parse telemetry data from .ibt
                telemetry_df = self.ibt_parser.parse_ibt_file(ibt_file)
                lap_stats = self.ibt_parser.extract_lap_statistics(telemetry_df)
                
                if lap_stats.empty:
                    print(f"   > Warning: No valid laps found in {ibt_file.name}")
                    continue

                # 3. Add setup data to *each lap*
                # We'll use the *average* valid lap as the performance metric
                
                # Filter out obvious outlier/warmup laps (e.g., > 20s at Bristol)
                lap_stats = lap_stats[lap_stats['lap_time'] < 20.0] 
                
                if lap_stats.empty:
                    print(f"   > Warning: No valid laps found in {ibt_file.name} after filtering.")
                    continue

                # Get the average of the valid laps
                avg_perf = lap_stats.mean().to_dict()
                
                # Get the best lap
                best_lap = lap_stats.loc[lap_stats['lap_time'].idxmin()].to_dict()
                
                # We'll use the "best_lap" for our analysis
                combined = {
                    **setup_data, 
                    **{f"best_{k}": v for k, v in best_lap.items()}
                }
                
                # Let's also add the "fastest_time" from the .ldx as a fallback
                if 'fastest_time' not in combined or pd.isna(combined['fastest_time']):
                    combined['fastest_time'] = combined.get('best_lap_time')

                all_data.append(combined)
            else:
                print(f"   > Warning: No matching .ibt file for {ldx_file.name}")

        
        df = pd.DataFrame(all_data)
        
        # Clean and prepare for agents
        df = self._prepare_for_agents(df)
        
        return df
    
    def _prepare_for_agents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for agent consumption"""

        # Ensure 'fastest_time' is the primary metric
        if 'best_lap_time' in df.columns:
            # Use the .ibt 'best_lap_time' if the .ldx 'fastest_time' is missing
            df['fastest_time'] = df['fastest_time'].fillna(df['best_lap_time'])

        if 'fastest_time' in df.columns:
            df = df[df['fastest_time'].notna()]
            df = df.sort_values('fastest_time')
            df['performance_rank'] = df['fastest_time'].rank()

        # Add performance delta from best
        if 'fastest_time' in df.columns and len(df) > 0:
            best_time = df['fastest_time'].min()
            df['time_delta_from_best'] = df['fastest_time'] - best_time

        return df