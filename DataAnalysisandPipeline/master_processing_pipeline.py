#
## Master Processing Pipeline for ADOT Crash Data
#
# This script performs the following steps:
#   1. Loads the raw ADOT crash dataset and filters for Wildlife-Vehicle Collisions (WVC).
#   2. Transforms the point-based crash data into a structured time-series format with regular temporal and spatial intervals.
#   3. Engineers sliding windows of historical data to create feature and target tensors suitable for training deep learning models (MLP, CNN, RNN).
# 

import pandas as pd
import numpy as np


def load_and_filter_adot_data(file_path):
    """
    Loads the ADOT crash dataset and filters for Wildlife-Vehicle Collisions (WVC).
    Expects columns: 'IncidentDate', 'IncidentDateTime', 'First_Harmful_Event', 'Milepost', etc.
    """
    df = pd.read_csv(file_path)

    df['Crash_DateTime'] = pd.to_datetime(
        df['IncidentDateTime'], 
        format="%m/%d/%Y %I:%M:%S %p", 
        errors='coerce'
    )
    
    # Filter strictly for Wildlife-Vehicle Collisions (WVC)
    # Code 21.0 = Wild Non-Game, Code 22.0 = Wild Game (Elk/Pronghorn from Code Tables)
    wvc_df = df[df['FirstHarmfulEvent'].isin([21.0, 22.0])].copy()
    
    return wvc_df

def build_master_time_series(wvc_df, freq='D', spatial_resolution_miles=5):
    """
    Transforms the point-based WVC crash data into a structured time-series format suitable for forecasting.
    - freq: Temporal frequency for aggregation (e.g., 'D' for daily)
    - spatial_resolution_miles: Width of each spatial segment (e.g., 5 miles)
    """
    
    # Create spatial segments along the route (e.g., 0-5mi, 5-10mi)
    wvc_df['Segment'] = (wvc_df['MPNum'] // spatial_resolution_miles) * spatial_resolution_miles
    
    # Generate an exhaustive grid of all possible (Time x Segment) combinations
    all_dates = pd.date_range(start='2021-01-01', end='2021-12-31', freq=freq)
    all_segments = sorted(wvc_df['Segment'].unique())
    
    multi_index = pd.MultiIndex.from_product([all_dates, all_segments], names=['Date', 'Segment'])
    master_grid = pd.DataFrame(index=multi_index).reset_index()
    
    # --- FIX HERE ---
    # Normalize timestamps to midnight to create a uniform day column while preserving datetime64 dtype
    wvc_df['Date'] = wvc_df['Crash_DateTime'].dt.normalize()
    
    # Aggregate actual crash counts by the uniform Day per segment
    counts = wvc_df.groupby(['Date', 'Segment']).size().reset_index(name='Crash_Count')
    # ----------------
    
    # Merge counts into the master grid, filling empty days/segments with 0 crashes
    master_df = pd.merge(master_grid, counts, on=['Date', 'Segment'], how='left')
    master_df['Crash_Count'] = master_df['Crash_Count'].fillna(0).astype(int)
    
    # Extract temporal features for potential use as exogenous variables in forecasting models
    master_df['DayOfWeek'] = master_df['Date'].dt.dayofweek
    master_df['Month'] = master_df['Date'].dt.month
    
    # Sine/Cosine transformations preserve temporal proximity (Dec 31 is close to Jan 1)
    master_df['Month_Sin'] = np.sin(2 * np.pi * master_df['Month']/12.0)
    master_df['Month_Cos'] = np.cos(2 * np.pi * master_df['Month']/12.0)
    
    # Pivot the table so rows = Time steps, and columns = Spatial Segments (Multi-point setup)
    ts_matrix = master_df.pivot(index='Date', columns='Segment', values='Crash_Count').fillna(0)
    
    return master_df, ts_matrix


def create_forecasting_tensors(data_matrix, exogenous_features=None, lookback=14, horizon=3):
    """
    Prepares shapes for MLP, CNN, and RNN networks.
    Supports: Univariate, Multivariate, Single-point, Multi-point, Single-step, Multi-step
    
    lookback (p): Days of history used to predict (e.g., 14 days)
    horizon (h): Days into the future to forecast (e.g., 3 days for multi-point/multi-step)
    """
    X, Y = [], []
    for t in range(len(data_matrix) - lookback - horizon + 1):

        x_window = data_matrix.iloc[t : t + lookback].values
        
        if exogenous_features is not None:
            x_exo = exogenous_features.iloc[t : t + lookback].values
            x_window = np.hstack((x_window, x_exo)) # Result: Multivariate shape
            
        y_window = data_matrix.iloc[t + lookback : t + lookback + horizon].values
        
        X.append(x_window)
        Y.append(y_window)
        
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    
    # File Path Definition
    adot_file_path = r"xxx.csv" # <-- UPDATE THIS PATH TO YOUR LOCAL ADOT CSV FILE
    
    print("\n" + "="*50)
    print("[*] STEP 1: Loading and filtering raw ADOT crash records...")
    print("="*50)
    
    try:
        wvc_data = load_and_filter_adot_data(adot_file_path)
        print(f"[+] Success! Extracted {len(wvc_data)} Wildlife-Vehicle Collision records.")
        
        print("\n" + "="*50)
        print("[*] STEP 2: Aggregating into regular spatial-temporal grid...")
        print("="*50)

        # Bins data daily ('D') and splits the highway into 5-mile segments
        master_flat_df, time_series_matrix = build_master_time_series(
            wvc_data, 
            freq='D', 
            spatial_resolution_miles=5
        )
        print(f"[+] Continuous time grid created.")
        print(f"[+] Master Matrix Shape (Days x Spatial Segments): {time_series_matrix.shape}")
        
        print("\n" + "="*50)
        print("[*] STEP 3: Engineering sliding windows for Deep Learning...")
        print("="*50)

        # Use a 14-day lookback window to predict the next 3 days
        X_tensor, Y_tensor = create_forecasting_tensors(
            time_series_matrix, 
            exogenous_features=None,  
            lookback=14, 
            horizon=3
        )
        print(f"[+] Deep Learning Tensors generated successfully!")
        print(f"[+] Feature Tensor Shape (Samples, Lookback, Segments): {X_tensor.shape}")
        print(f"[+] Target Tensor Shape  (Samples, Horizon, Segments) : {Y_tensor.shape}")
        
        # Save a quick copy to verify 
        time_series_matrix.to_csv("processed_wvc_timeseries.csv")
        print("\n[+] Saved processed matrix to processed_wvc_timeseries.csv")
        
    except KeyError as e:
        print(f"\\n[!] Column Mapping Error: The script couldn't find a required column: {e}")
        print("[*] Tip: Check if ADOT's headers are uppercase (e.g., 'Milepost' vs 'MILEPOST')")
    
    except FileNotFoundError:
        print(f"\\n[!] File Error: Could not find the CSV file at path:\\n{adot_file_path}")