from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import pandas as pd
from utils import preprocess_data, read_csv_with_metadata, plot_all_line_responses
from LSTM_from_PSTM import LineTransferMobility
import numpy as np
import plotly.graph_objects as go

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_data(
    request: Request,
    force_csv: UploadFile = File(...),
    other_csvs: list[UploadFile] = File(...),
    channel_distances: str = Form(...),
    impact_times: str = Form(...),
    train_length: float = Form(...),
    receiver_distances: str = Form(...),
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

    # Parse channel_distances and impact_times
    channel_distances_list = [float(d.strip()) for d in channel_distances.split(',')]
    impact_times_list = [float(t.strip()) for t in impact_times.split(',')]
    receiver_distances_list = [float(d.strip()) for d in receiver_distances.split(',')]

    if len(channel_distances_list) != len(vibration_csv_paths):
        return templates.TemplateResponse("results.html", {"request": request, "error": "Mismatch between number of channel distances and uploaded CSVs."})

    # Process reference CSV (force measurements)
    force_df = read_csv_with_metadata(force_csv_path)
    force_df.name = "force_df"
    force_measurements = preprocess_data(force_df, impact_times_list)
    force_measurements = {k: np.median(np.array(v)) for k, v in force_measurements.items()}
    pstm_measurements = {}
    channel_nums = [int(p[p.find('CH')+2]) for p in vibration_csv_paths]
    offset = min(channel_nums)
    # Process other CSVs (vibration measurements) and calculate transfer mobility
    for vib_channel_csv_path in vibration_csv_paths:
        ind = int(vib_channel_csv_path[vib_channel_csv_path.find('CH')+2]) - offset
        other_df = read_csv_with_metadata(vib_channel_csv_path)
        other_df.name = vib_channel_csv_path[-7:-3]
        vibration_measurements = preprocess_data(other_df, impact_times_list)

        # Perform subtraction (Vibration Level - Force Level)
        current_distance_measurements = {}
        for band_center, vib_level in vibration_measurements.items():
            force_level = force_measurements.get(band_center, [])
            if not np.any(np.isnan(vib_level)) and not np.any(np.isnan(force_level)):
                current_distance_measurements[band_center] = np.median(np.array(vib_level))
            else:
                current_distance_measurements[band_center] = np.nan # Or handle as appropriate
        
        pstm_measurements[channel_distances_list[ind]] = current_distance_measurements
    ltm_list = []
    for rd in receiver_distances_list:
        # Initialize LineTransferMobility
        ltm = LineTransferMobility(
            force_measurements=force_measurements,
            velocity_measurements=pstm_measurements,
            train_length=train_length,
            receiver_offset=rd,
            source_depth=source_depth
        )
        # Perform calculations and generate plots
        ltm.regress_point_sources()
        ltm.compute_all_line_responses()
        ltm_list.append(ltm)
    # Generate Plotly figures using LTM class methods
    plots = {}
    fig_pr = ltm_list[0].plot_point_regressions()
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

    plots["point_regressions"] = fig_pr.to_html(full_html=False, include_plotlyjs=False)
    # plots["force_measurements"] = fig_fm.to_html(full_html=False, include_plotlyjs=False)
    plots["measurements_level_vs_distance"] = fig_pstm_distance.to_html(full_html=False, include_plotlyjs=False)
    plots["measurements_level_vs_frequency"] = fig_pstm_frequency.to_html(full_html=False, include_plotlyjs=False)
    plots["all_line_responses"] = fig_lstms.to_html(full_html=False, include_plotlyjs=False)
    # Clean up temporary files
    os.remove(force_csv_path)
    for vib_channel_csv_path in vibration_csv_paths:
        os.remove(vib_channel_csv_path)
    os.rmdir(temp_dir)

    subtitle = f"train_length = {train_length}m, source_depth = {source_depth}m"
    return templates.TemplateResponse("results.html", {"request": request, "plots": plots, "subtitle": subtitle})

if __name__ == "__main__":
    if not os.path.exists("templates"):
        os.makedirs("templates")
    
    # Create a basic index.html for now
    with open("templates/index.html", "w") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vibration Impact Analysis</title>
        </head>
        <body>
            <h1>Upload Data for Vibration Impact Analysis</h1>
            <form action="/analyze" method="post" enctype="multipart/form-data">
                <label for="force_csv">Reference CSV:</label><br>
                <input type="file" id="force_csv" name="force_csv"><br><br>
                
                <label for="other_csvs">Other CSV Files (can upload multiple):</label><br>
                <input type="file" id="other_csvs" name="other_csvs" multiple><br><br>
                
                <label for="channel_distances">Channel Distances (comma-separated):</label><br>
                <input type="text" id="channel_distances" name="channel_distances"><br><br>
                
                <label for="impact_times">Impact Times (comma-separated):</label><br>
                <input type="text" id="impact_times" name="impact_times"><br><br>
                
                <label for="train_length">Train Length (m):</label><br>
                <input type="number" id="train_length" name="train_length" step="any"><br><br>
                
                <label for="receiver_distances">Receiver Distances (m):</label><br>
                <input type="text" id="receiver_distances" name="receiver_distances"><br><br>
                
                <label for="source_depth">Source Depth (m):</label><br>
                <input type="number" id="source_depth" name="source_depth" step="any" value="0.0"><br><br>
                
                <input type="submit" value="Analyze">
            </form>
        </body>
        </html>
        """)

    uvicorn.run(app, host="0.0.0.0", port=8000)
