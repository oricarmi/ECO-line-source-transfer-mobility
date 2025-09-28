import pandas as pd
import numpy as np
import os
import json
from conf import THIRD_OCTAVE_BANDS, IMPACT_SEARCH_RANGE

def get_exact_impact_times(data: pd.DataFrame, impact_times: list[float]) -> list[float]:
    impact_times_exact = []
    data["Time"] = pd.to_numeric(data["Time"]) # Ensure 'Time' is numeric
    delta_t = data["Time"].iloc[1] - data["Time"].iloc[0]

    for impact_time in impact_times:
        # Filter data within the search range
        filtered_data = data[(data["Time"] > impact_time - IMPACT_SEARCH_RANGE / delta_t) & (data["Time"] < impact_time + IMPACT_SEARCH_RANGE / delta_t)]
        
        if not filtered_data.empty:
            # Find the row with the maximum 'SUM(LIN)' within the filtered data
            argmax_index = filtered_data['SUM(LIN)'].idxmax()
            exact_impact_time = filtered_data.loc[argmax_index, "Time"]
            impact_times_exact.append(exact_impact_time)
        else:
            # If no data found in range, append the original impact_time or handle as error
            impact_times_exact.append(impact_time) # Or raise an error/warning

    return impact_times_exact

def preprocess_data(data: pd.DataFrame, exact_impact_times: list[float]) -> dict:
    """
    Preprocess the data to extract 1/3 octave band levels at exact impact times
    and return the average levels across all impacts.
    """
    all_band_levels_for_averaging = {band_center: [] for band_center in THIRD_OCTAVE_BANDS.values()}
    
    # Ensure 'Time' column is numeric
    data["Time"] = pd.to_numeric(data["Time"])

    # Convert all band columns to numeric
    for band_str in THIRD_OCTAVE_BANDS.keys():
        if band_str in data.columns:
            data[band_str] = pd.to_numeric(data[band_str], errors='coerce')

    # Filter data based on exact impact times and collect band levels
    for impact_time in exact_impact_times:
        impact_row = data[np.isclose(data["Time"], impact_time)]
        
        if not impact_row.empty:
            impact_row = impact_row.iloc[0] # Get the first matching row
            for band_str, band_center in THIRD_OCTAVE_BANDS.items():
                if band_str in impact_row and not pd.isna(impact_row[band_str]):
                    all_band_levels_for_averaging[band_center].append(impact_row[band_str])
        else:
            print(f"Warning: No data found for exact impact time {impact_time}")
            
    averaged_measurements = {}
    for band_center, levels in all_band_levels_for_averaging.items():
        if levels:
            averaged_measurements[band_center] = np.mean(levels)
        else:
            averaged_measurements[band_center] = np.nan # Handle cases where no data was collected for a band
    
    return averaged_measurements