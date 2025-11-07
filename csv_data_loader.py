"""
CSV Data Loader for Real Lap Data
Loads lap-by-lap telemetry data exported from .ibt files
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np


class CSVDataLoader:
    """
    Load and validate lap data from CSV exports
    Supports multiple CSV formats from common telemetry tools
    """

    # Expected columns for setup analysis
    REQUIRED_COLUMNS = [
        'lap_time',  # Must have lap times
    ]

    # Setup parameters we want to analyze
    SETUP_PARAMS = [
        'tire_psi_lf', 'tire_psi_rf', 'tire_psi_lr', 'tire_psi_rr',
        'cross_weight', 'track_bar_height_left',
        'spring_lf', 'spring_rf', 'spring_lr', 'spring_rr'
    ]

    # Telemetry metrics
    TELEMETRY_METRICS = [
        'avg_speed', 'max_speed', 'min_speed',
        'tire_temp_lf_avg', 'tire_temp_rf_avg', 'tire_temp_lr_avg', 'tire_temp_rr_avg',
        'lat_accel_max', 'brake_avg', 'throttle_avg'
    ]

    def __init__(self):
        self.data_paths = [
            Path('data/processed/bristol_lap_data.csv'),
            Path('data/processed/lap_data.csv'),
            Path('data/bristol_lap_data.csv'),
            Path('lap_data.csv'),
        ]
        self.ldx_path = Path('data/processed')

    def find_data_file(self) -> Optional[Path]:
        """Find the first available CSV data file"""
        for path in self.data_paths:
            if path.exists():
                return path
        return None

    def load_data(self, filepath: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """
        Load lap data from CSV file or .ldx files

        Args:
            filepath: Optional path to CSV file. If None, searches common locations.

        Returns:
            DataFrame with lap data, or None if no data found
        """

        # First try to load .ldx files (MoTeC format with setup data)
        ldx_data = self._load_ldx_files()
        if ldx_data is not None:
            return ldx_data

        # Fall back to CSV
        if filepath is None:
            filepath = self.find_data_file()

        if filepath is None:
            print("[WARNING]  No CSV or .ldx data files found")
            print(f"   CSV searched: {[str(p) for p in self.data_paths]}")
            print(f"   LDX searched: {self.ldx_path}/*.ldx")
            return None

        try:
            df = pd.read_csv(filepath)
            print(f"[OK] Loaded real data from: {filepath}")
            print(f"  {len(df)} laps, {len(df.columns)} columns")

            # Validate and prepare
            df = self._validate_and_prepare(df)

            return df

        except Exception as e:
            print(f"[ERROR] Error loading CSV: {e}")
            return None

    def _load_ldx_files(self) -> Optional[pd.DataFrame]:
        """Load data from .ldx files (MoTeC format)"""

        ldx_files = list(self.ldx_path.glob('*.ldx')) if self.ldx_path.exists() else []

        if not ldx_files:
            return None

        try:
            from telemetry_parser import TelemetryParser
            parser = TelemetryParser()

            all_sessions = []
            for ldx_file in ldx_files:
                data = parser.parse_ldx_file(ldx_file)
                all_sessions.append(data)

            df = pd.DataFrame(all_sessions)
            print(f"[OK] Loaded real data from {len(ldx_files)} .ldx files")
            print(f"  {len(df)} sessions from {df['venue'].iloc[0] if 'venue' in df.columns else 'unknown track'}")

            # Validate and prepare
            df = self._validate_and_prepare(df)

            return df

        except Exception as e:
            print(f"[ERROR] Error loading .ldx files: {e}")
            return None

    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data structure and prepare for analysis"""

        # Check for required columns
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                # Try to find alternative column names
                if 'fastest_time' in df.columns and col == 'lap_time':
                    df['lap_time'] = df['fastest_time']
                elif 'LapTime' in df.columns and col == 'lap_time':
                    df['lap_time'] = df['LapTime']
                else:
                    raise ValueError(f"Required column '{col}' not found in CSV")

        # Rename common variations to standard names
        column_mapping = {
            'session_id': 'session_id',
            'SessionID': 'session_id',
            'lap_number': 'lap_number',
            'LapNumber': 'lap_number',
            'lap_time': 'lap_time',
            'LapTime': 'lap_time',
            'fastest_time': 'lap_time',
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and old_name != new_name:
                df[new_name] = df[old_name]

        # Convert lap_time to numeric
        if 'lap_time' in df.columns:
            df['lap_time'] = pd.to_numeric(df['lap_time'], errors='coerce')

        # Remove invalid laps
        original_len = len(df)
        df = df[df['lap_time'].notna()]
        df = df[df['lap_time'] > 0]

        if len(df) < original_len:
            print(f"  Filtered out {original_len - len(df)} invalid laps")

        # Identify which setup parameters are available
        available_setup = [p for p in self.SETUP_PARAMS if p in df.columns]
        available_metrics = [m for m in self.TELEMETRY_METRICS if m in df.columns]

        print(f"  Setup parameters: {len(available_setup)} available")
        print(f"  Telemetry metrics: {len(available_metrics)} available")

        if len(available_setup) == 0:
            print("  [WARNING]  Warning: No setup parameters found in CSV")
            print("     Analysis will be limited to lap time trends only")

        # Sort by session and lap number if available
        if 'session_id' in df.columns and 'lap_number' in df.columns:
            df = df.sort_values(['session_id', 'lap_number'])
        elif 'lap_time' in df.columns:
            df = df.sort_values('lap_time')

        # Add performance metrics
        df = self._add_performance_metrics(df)

        return df

    def _add_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed performance metrics"""

        if 'lap_time' in df.columns and len(df) > 0:
            # Performance rank
            df['performance_rank'] = df['lap_time'].rank()

            # Delta from best lap
            best_time = df['lap_time'].min()
            df['time_delta_from_best'] = df['lap_time'] - best_time

            # Percentile
            df['lap_time_percentile'] = df['lap_time'].rank(pct=True) * 100

        # Add derived setup metrics if base parameters exist
        if all(col in df.columns for col in ['tire_psi_lf', 'tire_psi_rf']):
            df['tire_stagger_front'] = df['tire_psi_rf'] - df['tire_psi_lf']

        if all(col in df.columns for col in ['tire_psi_lr', 'tire_psi_rr']):
            df['tire_stagger_rear'] = df['tire_psi_rr'] - df['tire_psi_lr']

        if all(col in df.columns for col in ['spring_lf', 'spring_rf']):
            df['spring_ratio_front'] = df['spring_rf'] / df['spring_lf']

        return df

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics from the data"""

        stats = {
            'total_laps': len(df),
            'best_lap_time': float(df['lap_time'].min()) if 'lap_time' in df.columns else None,
            'worst_lap_time': float(df['lap_time'].max()) if 'lap_time' in df.columns else None,
            'average_lap_time': float(df['lap_time'].mean()) if 'lap_time' in df.columns else None,
            'std_lap_time': float(df['lap_time'].std()) if 'lap_time' in df.columns else None,
        }

        # Session count
        if 'session_id' in df.columns:
            stats['num_sessions'] = df['session_id'].nunique()

        # Available parameters
        stats['setup_params_available'] = [p for p in self.SETUP_PARAMS if p in df.columns]
        stats['telemetry_metrics_available'] = [m for m in self.TELEMETRY_METRICS if m in df.columns]

        return stats

    def prepare_for_ai_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data in the format expected by the AI agents
        Aggregates lap data to session level if needed
        """

        # Check if data is already at session level (has fastest_time but not lap_number)
        if 'fastest_time' in df.columns and 'lap_number' not in df.columns:
            # Already session-level data (like from .ldx files)
            print(f"  Using {len(df)} session-level records")
            return df

        # If we have lap-level data, aggregate to session level
        if 'session_id' in df.columns and 'lap_time' in df.columns:
            # Group by session and take the best lap from each session
            session_data = []

            for session_id in df['session_id'].unique():
                session_laps = df[df['session_id'] == session_id]

                # Get the best lap from this session
                best_lap_idx = session_laps['lap_time'].idxmin()
                best_lap = session_laps.loc[best_lap_idx].to_dict()

                # Also include average metrics
                avg_metrics = {}
                for col in session_laps.columns:
                    if col in self.TELEMETRY_METRICS:
                        avg_metrics[f'avg_{col}'] = session_laps[col].mean()

                best_lap.update(avg_metrics)
                best_lap['fastest_time'] = best_lap['lap_time']

                session_data.append(best_lap)

            df = pd.DataFrame(session_data)
            print(f"  Aggregated {len(df)} sessions from lap-level data")

        # Ensure 'fastest_time' column exists (required by agents)
        if 'lap_time' in df.columns and 'fastest_time' not in df.columns:
            df['fastest_time'] = df['lap_time']

        return df


def load_real_data() -> Optional[pd.DataFrame]:
    """
    Convenience function to load real data
    Returns None if no data available
    """
    loader = CSVDataLoader()
    return loader.load_data()


def get_data_for_analysis() -> pd.DataFrame:
    """
    Get data for AI analysis - real data if available, otherwise mock data
    """
    loader = CSVDataLoader()
    df = loader.load_data()

    if df is not None:
        df = loader.prepare_for_ai_analysis(df)
        return df
    else:
        print()
        print("="*70)
        print("  NO REAL DATA FOUND - Using mock data for demo")
        print("="*70)
        print()
        print("To use real data:")
        print("1. Export your .ibt files to CSV format")
        print("2. Save as: data/processed/bristol_lap_data.csv")
        print("3. See REAL_DATA_ANALYSIS.md for CSV format specification")
        print()

        # Return mock data
        from demo import generate_mock_data
        return generate_mock_data()


if __name__ == "__main__":
    # Test the loader
    print("Testing CSV Data Loader")
    print("="*70)
    print()

    loader = CSVDataLoader()
    df = loader.load_data()

    if df is not None:
        stats = loader.get_summary_statistics(df)
        print()
        print("Data Summary:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print()
        print("Sample data:")
        print(df.head())
    else:
        print()
        print("No CSV data found. To test with real data:")
        print("1. Create data/processed/bristol_lap_data.csv")
        print("2. Include columns: session_id, lap_time, tire_psi_lf, etc.")
        print("3. Run this script again")
