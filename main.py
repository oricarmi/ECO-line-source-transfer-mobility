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
from utils import preprocess_data, read_csv_with_metadata, plot_all_line_responses, plot_log_scale_measurements, read_fds_csv, read_receiver_excel, interpolate_lstm_for_receivers, create_receiver_lstm_tables, create_preview_plot
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
    receivers_data_excel: UploadFile | None = File(None),
    source_depth: float = Form(0.0),
    save_path: str = Form(""),
    units: str = Form("metric"),
    # New parameters for FDS and Excel files
    fds_csv: UploadFile | None = File(None),
    floor_resonance_frequencies: str = Form(""),
    db_amount: str = Form(""),
    # New parameters for preview flow
    force_csv_path: str = Form(None),
    vibration_csv_paths: str = Form(None),
    receivers_data_excel_path: str = Form(None),
    channel_distances_list: str = Form(None),
    impact_times_list: str = Form(None),
    receiver_distances_list: str = Form(None),
    impacts_to_remove: str = Form(""),
    fds_csv_path: str = Form(None)
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
        
        # Set FDS CSV path from preview flow
        fds_csv_path = fds_csv_path if fds_csv_path else None
        
        # Set Excel path from preview flow
        receivers_data_excel_path = receivers_data_excel_path if receivers_data_excel_path else None
        
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

        # Save FDS CSV file if provided
        fds_csv_path = None
        if fds_csv is not None and fds_csv.filename:
            fds_csv_path = os.path.join(temp_dir, fds_csv.filename)
            with open(fds_csv_path, "wb") as buffer:
                buffer.write(await fds_csv.read())

        # Parse channel_distances and impact_times
        channel_distances_list = [float(d.strip()) for d in channel_distances.split(',')]
        impact_times_list = [float(t.strip()) for t in impact_times.split(',')]
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
    
    # Generate log-scale plot with selected impacts only
    # Re-read the original data files for plotting
    try:
        force_df_original = read_csv_with_metadata(force_csv_path)
        vibration_dfs_original = []
        for vib_channel_csv_path in vibration_csv_paths:
            vib_df_original = read_csv_with_metadata(vib_channel_csv_path)
            vibration_dfs_original.append(vib_df_original)
        
        fig_log_scale = plot_log_scale_measurements(
            force_df_original, 
            vibration_dfs_original, 
            impact_times_list, 
            channel_distances_list, 
            units=units
        )
    except Exception as e:
        print(f"Error generating log scale plot: {e}")
        # Create a simple error plot
        fig_log_scale = go.Figure()
        fig_log_scale.add_annotation(text=f"Error generating log scale plot: {str(e)}", 
                                   xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig_log_scale.update_layout(title="Log Scale Plot - Error")
    project_name = os.path.basename(force_csv_path)[:os.path.basename(force_csv_path).find('1_3')-1]
    if save_path:
        fig_pr.write_image(os.path.join(save_path, f"{project_name}_point_regressions.png"))
        fig_fm.write_image(os.path.join(save_path, f"{project_name}_force_measurements.png"))
        fig_pstm_distance.write_image(os.path.join(save_path, f"{project_name}_pstm_vs_dist.png"))
        fig_pstm_frequency.write_image(os.path.join(save_path, f"{project_name}_pstm_vs_freq.png"))
        fig_lstms.write_image(os.path.join(save_path, f"{project_name}_lstms.png"))
        fig_log_scale.write_image(os.path.join(save_path, f"{project_name}_log_scale_measurements.png"))
        for ltm_instance in ltm_list:
            os.makedirs(save_path, exist_ok=True)
            # Save line responses to CSV
            line_responses_df = pd.DataFrame({
                'band_center_Hz': list(ltm_instance.lstm.keys()),
                'level_dB': list(ltm_instance.lstm.values())
            })
            line_responses_df.to_csv(os.path.join(save_path, f"{project_name}_lstm_receiver_{ltm_instance.receiver_offset}m.csv"), index=False)

    # Process FDS CSV and Excel files if provided
    receiver_tables = {}
    fds_data = None
    
    if fds_csv_path and os.path.exists(fds_csv_path):
        try:
            fds_data = read_fds_csv(fds_csv_path)
            print(f"FDS data loaded: {len(fds_data)} frequency bands")
        except Exception as e:
            print(f"Error reading FDS CSV: {e}")
    
    if receivers_data_excel_path and os.path.exists(receivers_data_excel_path):
        try:
            # Read receiver information from Excel file
            receivers = read_receiver_excel(receivers_data_excel_path)
            print(f"Found {len(receivers)} receivers")
            
            # Interpolate LSTM values for each receiver
            interpolated_receivers = interpolate_lstm_for_receivers(receivers, ltm_list, units)
            
            # Create HTML tables for each receiver
            receiver_tables = create_receiver_lstm_tables(interpolated_receivers, units)
            print(f"Created {len(receiver_tables)} receiver tables")
            
        except Exception as e:
            print(f"Error processing receiver Excel file: {e}")

    plots["log_scale_measurements"] = fig_log_scale.to_html(full_html=False, include_plotlyjs=False)
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
    if receivers_data_excel_path and os.path.exists(receivers_data_excel_path):
        os.remove(receivers_data_excel_path)
    if fds_csv_path and os.path.exists(fds_csv_path):
        os.remove(fds_csv_path)
    if os.path.exists(temp_dir):
        try:
            os.rmdir(temp_dir)
        except OSError:
            # Directory not empty, ignore
            pass

    title = f"Vibration Impact Analysis Results for {project_name}"
    subtitle = f"train_length = {train_length}m, source_depth = {source_depth}m."
    subtitle2 = f"Impact times: {impact_times_list}"
    return templates.TemplateResponse("results.html", {
        "request": request, 
        "plots": plots, 
        "title": title, 
        "subtitle": subtitle, 
        "subtitle2": subtitle2,
        "receiver_tables": receiver_tables,
        "fds_data": fds_data
    })

@app.post("/preview", response_class=HTMLResponse)
async def preview_data(
    request: Request,
    force_csv: UploadFile = File(...),
    other_csvs: list[UploadFile] = File(...),
    channel_distances: str = Form(...),
    impact_times: str = Form(...),
    train_length: float = Form(...),
    receiver_distances: str = Form(...),
    receivers_data_excel: UploadFile | None = File(None),
    source_depth: float = Form(0.0),
    save_path: str = Form(""),
    units: str = Form("metric"),
    # New parameters for FDS and Excel files
    fds_csv: UploadFile | None = File(None),
    floor_resonance_frequencies: str = Form(""),
    db_amount: str = Form("")
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

    # Save FDS CSV file if provided
    fds_csv_path = None
    if fds_csv is not None and fds_csv.filename:
        fds_csv_path = os.path.join(temp_dir, fds_csv.filename)
        with open(fds_csv_path, "wb") as buffer:
            buffer.write(await fds_csv.read())

    # Save Excel file if provided
    receivers_data_excel_path = None
    if receivers_data_excel is not None and receivers_data_excel.filename:
        receivers_data_excel_path = os.path.join(temp_dir, receivers_data_excel.filename)
        with open(receivers_data_excel_path, "wb") as buffer:
            buffer.write(await receivers_data_excel.read())

    # Parse inputs
    channel_distances_list = [float(d.strip()) for d in channel_distances.split(',')]
    impact_times_list = sorted([float(t.strip()) for t in impact_times.split(',')])
    receiver_distances_list = [float(d.strip()) for d in receiver_distances.split(',') if d.strip()]

    if len(channel_distances_list) != len(vibration_csv_paths):
        return templates.TemplateResponse("results.html", {"request": request, "error": "Mismatch between number of channel distances and uploaded CSVs."})

    # Process reference CSV (force measurements)
    force_df = read_csv_with_metadata(force_csv_path)
    force_df.name = "force_df"
    
    # Create preview plot using utils function
    plots = create_preview_plot(force_df, vibration_csv_paths, channel_distances_list, impact_times_list)
    
    # Store data for next step
    session_data = {
        "force_csv_path": force_csv_path,
        "vibration_csv_paths": vibration_csv_paths,
        "receivers_data_excel_path": receivers_data_excel_path,
        "channel_distances_list": channel_distances_list,
        "impact_times_list": impact_times_list,
        "receiver_distances_list": receiver_distances_list,
        "train_length": train_length,
        "source_depth": source_depth,
        "save_path": save_path,
        "units": units,
        "fds_csv_path": fds_csv_path,
        "floor_resonance_frequencies": floor_resonance_frequencies,
        "db_amount": db_amount
    }
    
    return templates.TemplateResponse("preview.html", {
        "request": request, 
        "plots": plots, 
        "impact_times": impact_times_list,
        "session_data": session_data
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
