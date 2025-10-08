from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import socket
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import preprocess_data, read_csv_with_metadata, plot_all_line_responses
from LSTM_from_PSTM import LineTransferMobility
from conf import THIRD_OCTAVE_BANDS

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    machine_ip = socket.gethostbyname(socket.gethostname())
    port = request.url.port
    app_url = f"http://{machine_ip}:{port}"
    return templates.TemplateResponse("index.html", {"request": request, "machine_ip": machine_ip, "app_url": app_url})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_data(
    request: Request,
    force_csv: UploadFile = File(None),
    other_csvs: list[UploadFile] = File(None),
    channel_distances: str = Form(None),
    impact_times: str = Form(None),
    train_length: float = Form(None),
    receiver_distances: str = Form(None),
    receiver_offsets_csv: UploadFile | None = File(None),
    source_depth: float = Form(0.0),
    save_path: str = Form(""),
    units: str = Form("metric"),
    # New parameters for preview flow
    force_csv_path: str = Form(None),
    vibration_csv_paths: str = Form(None),
    receiver_offsets_csv_path: str = Form(None),
    channel_distances_list: str = Form(None),
    impact_times_list: str = Form(None),
    receiver_distances_list: str = Form(None),
    impacts_to_remove: str = Form("")
):
    # Initialize temp_dir for cleanup
    temp_dir = "temp_uploads"
    
    # Determine if this is coming from preview or direct upload
    if force_csv_path:  # Coming from preview
        # Parse the stored data
        force_csv_path = force_csv_path
        vibration_csv_paths = vibration_csv_paths.split(',') if vibration_csv_paths else []
        channel_distances_list = [float(d.strip()) for d in channel_distances_list.split(',')] if channel_distances_list else []
        impact_times_list = [float(t.strip()) for t in impact_times_list.split(',')] if impact_times_list else []
        receiver_distances_list = [float(d.strip()) for d in receiver_distances_list.split(',')] if receiver_distances_list else []
        
        # Process impacts to remove
        if impacts_to_remove.strip():
            try:
                impacts_to_remove_list = [int(x.strip()) - 1 for x in impacts_to_remove.split(',') if x.strip()]  # Convert to 0-based indexing
                # Remove the specified impacts
                impact_times_list = [time for i, time in enumerate(impact_times_list) if i not in impacts_to_remove_list]
            except ValueError:
                return templates.TemplateResponse("results.html", {"request": request, "error": "Invalid impact numbers. Please use comma-separated integers."})
    else:  # Direct upload flow
        # Save uploaded files temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)

        force_csv_path = os.path.join(temp_dir, force_csv.filename)
        with open(force_csv_path, "wb") as buffer:
            buffer.write(await force_csv.read())

        vibration_csv_paths = []
        for csv_file in other_csvs:
            file_path = os.path.join(temp_dir, csv_file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await csv_file.read())
            vibration_csv_paths.append(file_path)

        # Parse channel_distances and impact_times
        channel_distances_list = [float(d.strip()) for d in channel_distances.split(',')]
        impact_times_list = [float(t.strip()) for t in impact_times.split(',')]

        # Receiver distances: prefer CSV if provided, else use free text
        receiver_offsets_csv_path = None
        if receiver_offsets_csv is not None and receiver_offsets_csv.filename:
            receiver_offsets_csv_path = os.path.join(temp_dir, receiver_offsets_csv.filename)
            with open(receiver_offsets_csv_path, "wb") as buffer:
                buffer.write(await receiver_offsets_csv.read())
            try:
                df_offsets = pd.read_csv(receiver_offsets_csv_path)
            except Exception as e:
                # Clean up before returning error
                os.remove(force_csv_path)
                for vib_channel_csv_path in vibration_csv_paths:
                    os.remove(vib_channel_csv_path)
                if receiver_offsets_csv_path and os.path.exists(receiver_offsets_csv_path):
                    os.remove(receiver_offsets_csv_path)
                os.rmdir(temp_dir)
                return templates.TemplateResponse("results.html", {"request": request, "error": f"Failed to read receiver offsets CSV: {e}"})

            if "receiver_offset" not in df_offsets.columns:
                # Clean up before returning error
                os.remove(force_csv_path)
                for vib_channel_csv_path in vibration_csv_paths:
                    os.remove(vib_channel_csv_path)
                if receiver_offsets_csv_path and os.path.exists(receiver_offsets_csv_path):
                    os.remove(receiver_offsets_csv_path)
                os.rmdir(temp_dir)
                return templates.TemplateResponse("results.html", {"request": request, "error": "Receiver offsets CSV must have a column named 'receiver_offset'."})

            try:
                receiver_distances_list = (
                    df_offsets["receiver_offset"].dropna().astype(float).tolist()
                )
            except Exception as e:
                # Clean up before returning error
                os.remove(force_csv_path)
                for vib_channel_csv_path in vibration_csv_paths:
                    os.remove(vib_channel_csv_path)
                if receiver_offsets_csv_path and os.path.exists(receiver_offsets_csv_path):
                    os.remove(receiver_offsets_csv_path)
                os.rmdir(temp_dir)
                return templates.TemplateResponse("results.html", {"request": request, "error": f"Invalid values in 'receiver_offset' column: {e}"})
        else:
            receiver_distances_list = [float(d.strip()) for d in receiver_distances.split(',') if d.strip()]

    if len(channel_distances_list) != len(vibration_csv_paths):
        return templates.TemplateResponse("results.html", {"request": request, "error": "Mismatch between number of channel distances and uploaded CSVs."})

    # Check if files exist (especially important for preview flow)
    if not os.path.exists(force_csv_path):
        return templates.TemplateResponse("results.html", {"request": request, "error": f"Force CSV file not found: {force_csv_path}. Current working directory: {os.getcwd()}"})
    
    for vib_path in vibration_csv_paths:
        if not os.path.exists(vib_path):
            return templates.TemplateResponse("results.html", {"request": request, "error": f"Vibration CSV file not found: {vib_path}. Current working directory: {os.getcwd()}"})

    # Process reference CSV (force measurements)
    force_df = read_csv_with_metadata(force_csv_path)
    force_df.name = "force_df"
    force_measurements = preprocess_data(force_df, impact_times_list)
    force_measurements = {k: np.median(np.array(v)) for k, v in force_measurements.items()}
    pstm_measurements = {}
    pstm_measurements_iqr = {}
    channel_nums = [int(p[p.find('CH')+2]) for p in vibration_csv_paths]
    offset = min(channel_nums)
    # Process other CSVs (vibration measurements) and calculate transfer mobility
    for vib_channel_csv_path in vibration_csv_paths:
        ind = int(vib_channel_csv_path[vib_channel_csv_path.find('CH')+2]) - offset
        vib_df = read_csv_with_metadata(vib_channel_csv_path)
        vib_df.name = vib_channel_csv_path[-7:-3]
        vibration_measurements = preprocess_data(vib_df, impact_times_list)

        # Perform subtraction (Vibration Level - Force Level)
        current_distance_measurements_median = {}
        current_distance_measurements_iqr = {}
        for band_center, vib_level in vibration_measurements.items():
            force_level = force_measurements.get(band_center, [])
            if not np.any(np.isnan(vib_level)) and not np.any(np.isnan(force_level)):
                # if band_center == 4:
                #     print(f"vib_level for {band_center} Hz channel {ind} at distance {channel_distances_list[ind]}m: {np.array(vib_level)}")
                current_distance_measurements_median[band_center] = np.mean(np.array(vib_level))
                current_distance_measurements_iqr[band_center] = np.std(np.array(vib_level))#np.percentile(np.array(vib_level), 75) - np.percentile(np.array(vib_level), 25)
            else:
                current_distance_measurements_median[band_center] = np.nan # Or handle as appropriate
                current_distance_measurements_iqr[band_center] = np.nan # Or handle as appropriate
        
        pstm_measurements[channel_distances_list[ind]] = current_distance_measurements_median
        pstm_measurements_iqr[channel_distances_list[ind]] = current_distance_measurements_iqr
    ltm_list = []
    for rd in receiver_distances_list:
        # Initialize LineTransferMobility
        ltm = LineTransferMobility(
            force_measurements=force_measurements,
            velocity_measurements=pstm_measurements,
            velocity_measurements_iqr=pstm_measurements_iqr,
            train_length=train_length,
            receiver_offset=rd,
            source_depth=source_depth
        )
        # Perform calculations and generate plots
        ltm.regress_point_sources()
        ltm.compute_lstms_all_freqs()
        ltm_list.append(ltm)
    # Generate Plotly figures using LTM class methods
    plots = {}
    fig_pr = ltm_list[0].plot_point_regressions(units=units)
    fig_fm = ltm_list[0].plot_force_measurements()
    fig_pstm_distance = ltm_list[0].plot_pstm_level_vs_distance(units=units)
    fig_pstm_frequency = ltm_list[0].plot_pstm_level_vs_frequency(units=units)
    fig_lstms = plot_all_line_responses(ltm_list, units=units)
    if save_path:
        project_name = os.path.basename(force_csv_path)[:os.path.basename(force_csv_path).find('Velocity')]
        fig_pr.write_image(os.path.join(save_path, f"{project_name}_point_regressions.png"))
        fig_fm.write_image(os.path.join(save_path, f"{project_name}_force_measurements.png"))
        fig_pstm_distance.write_image(os.path.join(save_path, f"{project_name}_pstm_vs_dist.png"))
        fig_pstm_frequency.write_image(os.path.join(save_path, f"{project_name}_pstm_vs_freq.png"))
        fig_lstms.write_image(os.path.join(save_path, f"{project_name}_lstms.png"))
        for ltm_instance in ltm_list:
            os.makedirs(save_path, exist_ok=True)
            # Save line responses to CSV
            line_responses_df = pd.DataFrame({
                'band_center_Hz': list(ltm_instance.lstm.keys()),
                'level_dB': list(ltm_instance.lstm.values())
            })
            line_responses_df.to_csv(os.path.join(save_path, f"{project_name}_lstm_receiver_{ltm_instance.receiver_offset}m.csv"), index=False)

    plots["force_measurements"] = fig_fm.to_html(full_html=False, include_plotlyjs=False)
    plots["measurements_level_vs_distance"] = fig_pstm_distance.to_html(full_html=False, include_plotlyjs=False)
    plots["point_regressions"] = fig_pr.to_html(full_html=False, include_plotlyjs=False)
    plots["measurements_level_vs_frequency"] = fig_pstm_frequency.to_html(full_html=False, include_plotlyjs=False)
    plots["all_line_responses"] = fig_lstms.to_html(full_html=False, include_plotlyjs=False)
    # Clean up temporary files
    if os.path.exists(force_csv_path):
        os.remove(force_csv_path)
    for vib_channel_csv_path in vibration_csv_paths:
        if os.path.exists(vib_channel_csv_path):
            os.remove(vib_channel_csv_path)
    if receiver_offsets_csv_path and os.path.exists(receiver_offsets_csv_path):
        os.remove(receiver_offsets_csv_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

    subtitle = f"train_length = {train_length}m, source_depth = {source_depth}m"
    return templates.TemplateResponse("results.html", {"request": request, "plots": plots, "subtitle": subtitle})

@app.post("/preview", response_class=HTMLResponse)
async def preview_data(
    request: Request,
    force_csv: UploadFile = File(...),
    other_csvs: list[UploadFile] = File(...),
    channel_distances: str = Form(...),
    impact_times: str = Form(...),
    train_length: float = Form(...),
    receiver_distances: str = Form(...),
    receiver_offsets_csv: UploadFile | None = File(None),
    source_depth: float = Form(0.0),
    save_path: str = Form(""),
    units: str = Form("metric")
):
    # Save uploaded files temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    force_csv_path = os.path.join(temp_dir, force_csv.filename)
    with open(force_csv_path, "wb") as buffer:
        buffer.write(await force_csv.read())

    vibration_csv_paths = []
    for csv_file in other_csvs:
        file_path = os.path.join(temp_dir, csv_file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await csv_file.read())
        vibration_csv_paths.append(file_path)

    # Parse inputs
    channel_distances_list = [float(d.strip()) for d in channel_distances.split(',')]
    impact_times_list = sorted([float(t.strip()) for t in impact_times.split(',')])

    # Receiver distances: prefer CSV if provided, else use free text
    receiver_offsets_csv_path = None
    if receiver_offsets_csv is not None and receiver_offsets_csv.filename:
        receiver_offsets_csv_path = os.path.join(temp_dir, receiver_offsets_csv.filename)
        with open(receiver_offsets_csv_path, "wb") as buffer:
            buffer.write(await receiver_offsets_csv.read())
        try:
            df_offsets = pd.read_csv(receiver_offsets_csv_path)
        except Exception as e:
            # Clean up before returning error
            os.remove(force_csv_path)
            for vib_channel_csv_path in vibration_csv_paths:
                os.remove(vib_channel_csv_path)
            if receiver_offsets_csv_path and os.path.exists(receiver_offsets_csv_path):
                os.remove(receiver_offsets_csv_path)
            os.rmdir(temp_dir)
            return templates.TemplateResponse("results.html", {"request": request, "error": f"Failed to read receiver offsets CSV: {e}"})

        if "receiver_offset" not in df_offsets.columns:
            # Clean up before returning error
            os.remove(force_csv_path)
            for vib_channel_csv_path in vibration_csv_paths:
                os.remove(vib_channel_csv_path)
            if receiver_offsets_csv_path and os.path.exists(receiver_offsets_csv_path):
                os.remove(receiver_offsets_csv_path)
            os.rmdir(temp_dir)
            return templates.TemplateResponse("results.html", {"request": request, "error": "Receiver offsets CSV must have a column named 'receiver_offset'."})

        try:
            receiver_distances_list = (
                df_offsets["receiver_offset"].dropna().astype(float).tolist()
            )
        except Exception as e:
            # Clean up before returning error
            os.remove(force_csv_path)
            for vib_channel_csv_path in vibration_csv_paths:
                os.remove(vib_channel_csv_path)
            if receiver_offsets_csv_path and os.path.exists(receiver_offsets_csv_path):
                os.remove(receiver_offsets_csv_path)
            os.rmdir(temp_dir)
            return templates.TemplateResponse("results.html", {"request": request, "error": f"Invalid values in 'receiver_offset' column: {e}"})
    else:
        receiver_distances_list = [float(d.strip()) for d in receiver_distances.split(',') if d.strip()]

    if len(channel_distances_list) != len(vibration_csv_paths):
        return templates.TemplateResponse("results.html", {"request": request, "error": "Mismatch between number of channel distances and uploaded CSVs."})

    # Process reference CSV (force measurements)
    force_df = read_csv_with_metadata(force_csv_path)
    force_df.name = "force_df"
    
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
    
    # Store data for next step
    session_data = {
        "force_csv_path": force_csv_path,
        "vibration_csv_paths": vibration_csv_paths,
        "receiver_offsets_csv_path": receiver_offsets_csv_path,
        "channel_distances_list": channel_distances_list,
        "impact_times_list": impact_times_list,
        "receiver_distances_list": receiver_distances_list,
        "train_length": train_length,
        "source_depth": source_depth,
        "save_path": save_path,
        "units": units
    }
    
    return templates.TemplateResponse("preview.html", {
        "request": request, 
        "plots": plots, 
        "impact_times": impact_times_list,
        "session_data": session_data
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
