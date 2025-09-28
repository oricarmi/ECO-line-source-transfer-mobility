import pandas as pd
import numpy as np
import os
import json
from conf import THIRD_OCTAVE_BANDS, IMPACT_SEARCH_RANGE

def get_exact_impact_times(data: pd.DataFrame, impact_times: list[float]) -> list[float]:
    impact_times_exact = []
    for impact_time in impact_times:
        # Ensure 'Time' column is numeric for comparison
        data["Time"] = pd.to_numeric(data["Time"])
        delta_t = data["Time"].iloc[1] - data["Time"].iloc[0]
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

def preprocess_data(data: pd.DataFrame, force_measurements, exact_impact_times: list[float]) -> dict:
    """
    Preprocess the data to extract 1/3 octave band levels at exact impact times
    and return the average levels across all impacts.
    """
    measurements = {}
    # Ensure 'Time' column is numeric
    data = data.iloc[1:, :]
    data2 = (data[data["Time"].isin(exact_impact_times)] - force_measurements[force_measurements["Time"].isin(exact_impact_times)]).mean()
    averaged_measurements = {}
    for band_str, band in THIRD_OCTAVE_BANDS.items():
        averaged_measurements[band] = data2[band_str]

    return averaged_measurements