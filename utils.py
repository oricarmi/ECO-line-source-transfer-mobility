import pandas as pd
import numpy as np
import os
import json
from io import StringIO
from conf import THIRD_OCTAVE_BANDS, IMPACT_SEARCH_RANGE, y_axis_titles_lstm, metric_to_empirial_lstm
import plotly.graph_objs as go
import numpy as np

def read_csv_with_metadata(filename):
    # read the raw csv from samurai and return a pandas dataframe
    # Read the entire file as text
    with open(filename, 'r', encoding='cp1252') as file:
        lines = file.readlines()
    
    # Find the line that starts with "Time"
    csv_start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Time,'):
            csv_start_index = i
            break
    
    if csv_start_index is None:
        raise ValueError("Could not find 'Time' header row")
    
    # Join the CSV portion and read with pandas
    csv_data = ''.join(lines[csv_start_index:])
    df = pd.read_csv(StringIO(csv_data))
    return df.iloc[1:, :]



def get_exact_impact_time(data: pd.DataFrame, impact_time: float, freq, delta_t = 1.0) -> float:
    # Filter data within the search range
    filtered_data = data[(data["Time"] > impact_time - IMPACT_SEARCH_RANGE / delta_t) & (data["Time"] < impact_time + IMPACT_SEARCH_RANGE / delta_t)]
    if not filtered_data.empty:
        # Find the row with the maximum rms in the freq bin within the time window around it
        argmax_index = int(filtered_data[freq].idxmax())
        return filtered_data.loc[argmax_index, freq]
    print("Warning: filtered data was empty")
    return filtered_data.loc[impact_time, freq]


def preprocess_data(data: pd.DataFrame, impact_times: list[float]) -> dict:
    """
    Preprocess the data to extract max value in a window of 5 seconds around user recorded impact time for each band
    """
    all_band_levels_for_averaging = {band_center: [] for band_center in THIRD_OCTAVE_BANDS.values()}
    
    # Ensure 'Time' column is numeric
    data["Time"] = pd.to_numeric(data["Time"])
    delta_t = data["Time"].iloc[1] - data["Time"].iloc[0]
    
    # Filter data based on exact impact times and collect band level
    for band_str, band_center in THIRD_OCTAVE_BANDS.items():
        data[band_str] = pd.to_numeric(data[band_str], errors='coerce') # Convert all band columns to numeric
        for impact_time in impact_times:
            max_val_around_impact = get_exact_impact_time(data[["Time", band_str]], impact_time, band_str, delta_t)
            all_band_levels_for_averaging[band_center].append(max_val_around_impact)    
    return all_band_levels_for_averaging


def plot_all_line_responses(all_ltm_objs: list, units="metric") -> go.Figure:
    fig = go.Figure()
    for ltm in all_ltm_objs:
        vals = np.array(list(ltm.lstm.values()))
        if units == "imperial":
            vals += metric_to_empirial_lstm
        fig.add_trace(go.Scatter(x=ltm.band_centers, y=list(vals), mode='markers+lines', name=f"{ltm.receiver_offset} m"))
    fig.update_layout(
        title="Line Source Transfer Mobility for Different Receiver Distances",
        xaxis_title="Frequency (Hz)",
        yaxis_title=y_axis_titles_lstm[units],
        xaxis=dict(type="log")
    )
    return fig

def exponential_decay(distance, A, alpha, C):
    return A * np.exp(-alpha * distance) + C

def plot_log_scale_measurements(force_df, vibration_dfs, impact_times, channel_distances, units="metric"):
    """
    Create a log-scale plot similar to preview but with only chosen impacts.
    Force data is zeroed outside IMPACT_SEARCH_RANGE around chosen impact times.
    """
    from conf import IMPACT_SEARCH_RANGE, V_Ref, F_Ref, y_axis_titles_vib
    import pandas as pd
    
    fig = go.Figure()
    
    # Process force data - zero values outside IMPACT_SEARCH_RANGE around chosen impacts
    if "SUM(LIN)" in force_df.columns:
        # Convert to numeric and drop any rows with NaN values
        force_df["SUM(LIN)"] = pd.to_numeric(force_df["SUM(LIN)"], errors='coerce')
        force_df["Time"] = pd.to_numeric(force_df["Time"], errors='coerce')
        
        # Drop rows where Time or SUM(LIN) couldn't be converted to numeric
        force_df = force_df.dropna(subset=['Time', 'SUM(LIN)'])
        
        # Create a copy of force data and zero values outside impact ranges
        force_data_processed = force_df["SUM(LIN)"].copy()
        
        # # Set all values to a small value (1e-10) initially
        force_data_processed[:] = 1e-10
        
        # # For each chosen impact time, keep values within IMPACT_SEARCH_RANGE
        for impact_time in impact_times:
            mask = (force_df["Time"] >= float(impact_time) - IMPACT_SEARCH_RANGE/2) & \
                   (force_df["Time"] <= float(impact_time) + IMPACT_SEARCH_RANGE/2)
            force_data_processed[mask] = force_df["SUM(LIN)"][mask]
        
        # Convert to dB relative to F_Ref
        force_db = 20 * np.log10(np.maximum(force_data_processed, 1e-10) / F_Ref)
        
        fig.add_trace(go.Scatter(
            x=force_df["Time"], 
            y=force_db.to_list(),
            mode='lines', 
            name="Force SUM(LIN) (dB)",
            yaxis='y2',
            line=dict(color='blue')
        ))
    
    # Add vibration data (left y-axis) - only for chosen impacts
    colormap = ["brown", "red", "orange", "pink", "yellow", "purple", "brown", "green", "cyan", "magenta"]
    for i, (vib_df, distance) in enumerate(zip(vibration_dfs, channel_distances)):
        if "SUM(LIN)" in vib_df.columns:
            vib_df["SUM(LIN)"] = pd.to_numeric(vib_df["SUM(LIN)"], errors='coerce')
            # Convert to dB relative to V_Ref
            vib_db = 20 * np.log10(np.maximum(vib_df["SUM(LIN)"], 1e-10) / V_Ref)
            fig.add_trace(go.Scatter(
                x=vib_df["Time"], 
                y=vib_db.to_list(),
                mode='lines', 
                name=f"Vib CH{i+1} ({distance}m) SUM(LIN) dB",
                yaxis='y',
                line=dict(color=colormap[i % len(colormap)])
            ))
    
    # Add impact time lines
    for i, impact_time in enumerate(impact_times):
        fig.add_vline(x=float(impact_time), line_dash="dash", line_color="black", 
                      annotation_text=f"Impact {i+1}", annotation_position="top")
    
    fig.update_layout(
        title="Force and Vibration Measurements (Log Scale) - Selected Impacts Only",
        xaxis_title="Time (s)",
        yaxis=dict(
            title=y_axis_titles_vib[units],
            side="left",
            title_font=dict(color="red"),
            tickfont=dict(color="red")
        ),
        yaxis2=dict(
            title="Force Level (dB rel 1N)",
            side="right",
            overlaying="y",
            title_font=dict(color="blue"),
            tickfont=dict(color="blue")
        ),
        height=600
    )
    
    return fig