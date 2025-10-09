import pandas as pd
import numpy as np
import os
import json
from io import StringIO
from conf import THIRD_OCTAVE_BANDS, IMPACT_SEARCH_RANGE, y_axis_titles_lstm, metric_to_empirial_lstm
import plotly.graph_objs as go
import numpy as np
import typing

def read_csv_with_metadata(filename) -> pd.DataFrame:
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

def read_fds_csv(fds_csv_path) -> dict:
    """
    Read FDS CSV file with 1/3 octave spectrum columns.
    Returns a dictionary with frequency bands as keys and force density values as values.
    """
    
    # Read the CSV file
    df = pd.read_csv(fds_csv_path, header=None)
    
    # The first row contains the frequency band names
    freq_bands = df.iloc[0].tolist()
    
    # The second row contains the force density values
    force_density_values = df.iloc[1].tolist()
    
    # Create dictionary mapping frequency bands to values
    fds_data = {}
    for band, value in zip(freq_bands, force_density_values):
        if pd.notna(band) and pd.notna(value):
            fds_data[band] = float(value)
    
    return fds_data

def read_receiver_excel(excel_path) -> list[dict]:
    """
    Read Excel file, set 3rd row as columns, and extract receiver information.
    Returns a list of dictionaries with receiver information.
    """
    import pandas as pd
    
    # Read Excel file, skipping first 2 rows and using 3rd row as header
    df = pd.read_excel(excel_path, sheet_name='Receivers_Vib', header=2)
    
    receivers = []
    for _, row in df.iterrows():
        if pd.notna(row.get('S.N', '')) and pd.notna(row.get('Building', '')):
            receiver_info = {
                'name': f"{row['S.N']} {row['Building']}",
                'distance': float(row['Horizontal distance to the nearest rail axis, m']),
                'row_data': row.to_dict()
            }
            receivers.append(receiver_info)
    
    return receivers

def interpolate_lstm_for_receivers(receivers, ltm_list, units="metric") -> list[dict]:
    """
    Linear interpolate LSTM values for each receiver distance.
    Returns a list of dictionaries with receiver names and interpolated LSTM values.
    """
    from scipy.interpolate import interp1d
    
    # Extract distances and LSTM values from ltm_list
    distances = [ltm.receiver_offset for ltm in ltm_list]
    
    # Get frequency bands from the first LTM object
    freq_bands = list(ltm_list[0].lstm.keys())
    
    # For each frequency band, create interpolation function
    interpolated_receivers = []
    
    for receiver in receivers:
        receiver_lstm = {}
        
        for freq in freq_bands:
            # Extract LSTM values for this frequency across all distances
            lstm_values = []
            for ltm in ltm_list:
                lstm_val = ltm.lstm[freq]
                if units == "imperial":
                    lstm_val += metric_to_empirial_lstm
                lstm_values.append(lstm_val)
            
            # Create interpolation function
            interp_func = interp1d(distances, lstm_values, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
            
            # Interpolate for this receiver's distance
            receiver_lstm[freq] = float(interp_func(receiver['distance']))
        
        interpolated_receivers.append({
            'name': receiver['name'],
            'distance': receiver['distance'],
            'lstm_values': receiver_lstm,
            'row_data': receiver['row_data']
        })
    
    return interpolated_receivers

def create_receiver_lstm_tables(interpolated_receivers, units="metric") -> dict:
    """
    Create HTML tables for each receiver showing LSTM values.
    Returns a dictionary with receiver names as keys and HTML table strings as values.
    """
    from conf import THIRD_OCTAVE_BANDS
    
    tables = {}
    
    for receiver in interpolated_receivers:
        # Create HTML table with frequencies as columns
        table_html = f"""
        <div class="receiver-table">
            <h3>Receiver: {receiver['name']} (Distance: {receiver['distance']:.1f}m)</h3>
            <table border="1" style="border-collapse: collapse; margin: 10px 0;">
                <thead>
                    <tr>
                        <th>LSTM ({y_axis_titles_lstm[units]})</th>
        """
        
        # Add frequency columns to header
        for freq_band_str, freq_value in THIRD_OCTAVE_BANDS.items():
            if freq_value in receiver['lstm_values']:
                table_html += f"<th>{freq_band_str}</th>"
        
        table_html += """
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Value</td>
        """
        
        # Add LSTM values as row data
        for freq_band_str, freq_value in THIRD_OCTAVE_BANDS.items():
            if freq_value in receiver['lstm_values']:
                lstm_value = receiver['lstm_values'][freq_value]
                table_html += f"<td>{lstm_value:.2f}</td>"
        
        table_html += """
                    </tr>
                </tbody>
            </table>
        </div>
        """
        
        tables[receiver['name']] = table_html
    
    return tables

def create_preview_plot(force_df, vibration_csv_paths, channel_distances_list, impact_times_list) -> dict:
    """
    Create the combined preview plot showing force and vibration measurements with impact times.
    """
    import plotly.graph_objects as go
    import pandas as pd
    
    # Initialize plots dictionary
    plots = {}
    
    # Create combined plot with dual y-axes
    fig_combined = go.Figure()
    
    # Add force data (right y-axis) - SUM_LIN only
    if "SUM(LIN)" in force_df.columns:
        force_df["SUM(LIN)"] = pd.to_numeric(force_df["SUM(LIN)"], errors='coerce')
        fig_combined.add_trace(go.Scatter(
            x=force_df["Time"], 
            y=list(force_df["SUM(LIN)"]), 
            mode='lines', 
            name="Force SUM(LIN)",
            yaxis='y2',
            line=dict(color='blue')
        ))
    
    # Add vibration data (left y-axis) - SUM_LIN only
    colormap = ["red", "green", "orange", "purple", "brown", "pink", "gray", "yellow", "cyan", "magenta"]
    for i, vib_channel_csv_path in enumerate(vibration_csv_paths):
        vib_df = read_csv_with_metadata(vib_channel_csv_path)
        
        # Extract channel number from filename
        ch_start = vib_channel_csv_path.find('CH')
        if ch_start != -1:
            # Find the number after CH
            ch_num = ""
            for j in range(ch_start + 2, len(vib_channel_csv_path)):
                if vib_channel_csv_path[j].isdigit():
                    ch_num += vib_channel_csv_path[j]
                else:
                    break
            channel_name = f"CH{ch_num}" if ch_num else f"CH{i+1}"
        else:
            channel_name = f"CH{i+1}"
        
        if "SUM(LIN)" in vib_df.columns:
            vib_df["SUM(LIN)"] = pd.to_numeric(vib_df["SUM(LIN)"], errors='coerce')
            fig_combined.add_trace(go.Scatter(
                x=vib_df["Time"], 
                y=list(vib_df["SUM(LIN)"]), 
                mode='lines', 
                name=f"Vib {channel_name} SUM(LIN) ({channel_distances_list[i]}m)",
                yaxis='y',
                line=dict(color=colormap[i])
            ))
    
    # Add impact time lines
    for i, impact_time in enumerate(impact_times_list):
        fig_combined.add_vline(x=impact_time, line_dash="dash", line_color="black", 
                              annotation_text=f"Impact {i+1}", annotation_position="top")
    
    fig_combined.update_layout(
        title="Force and Vibration Measurements with Impact Times",
        xaxis_title="Time (s)",
        yaxis=dict(
            title="Vibration Level (m/s)",
            side="left",
            title_font=dict(color="red"),
            tickfont=dict(color="red")
        ),
        yaxis2=dict(
            title="Force Level (N)",
            side="right",
            overlaying="y",
            title_font=dict(color="blue"),
            tickfont=dict(color="blue")
        ),
        height=600
    )
    
    plots["combined_preview"] = fig_combined.to_html(full_html=False, include_plotlyjs=False)
    return plots

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