import pandas as pd
import numpy as np
import os
import json
from io import StringIO
from conf import THIRD_OCTAVE_BANDS, IMPACT_SEARCH_RANGE
import plotly.graph_objs as go

def read_csv_with_metadata(filename):
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


def plot_all_line_responses(all_ltm_objs: list) -> go.Figure:
    fig = go.Figure()
    for ltm in all_ltm_objs:
        fig.add_trace(go.Scatter(x=ltm.band_centers, y=list(ltm.line_responses.values()), mode='markers+lines', name=f"{ltm.receiver_offset} m"))
    fig.update_layout(
        title="Line Responses for Different Receiver Distances",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Level (dB rel 50nm/sec / (N/sqrt(m)) )",
        xaxis=dict(type="log")
    )
    return fig