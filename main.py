from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import pandas as pd
from utils import get_exact_impact_times, preprocess_data, read_csv_with_metadata
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
    reference_csv: UploadFile = File(...),
    other_csvs: list[UploadFile] = File(...),
    channel_distances: str = Form(...),
    impact_times: str = Form(...),
    train_length: float = Form(...),
    receiver_distance: float = Form(...),
    source_depth: float = Form(0.0)
):
    # Save uploaded files temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    ref_csv_path = os.path.join(temp_dir, reference_csv.filename)
    with open(ref_csv_path, "wb") as buffer:
        buffer.write(await reference_csv.read())

    other_csv_paths = []
    for csv_file in other_csvs:
        file_path = os.path.join(temp_dir, csv_file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await csv_file.read())
        other_csv_paths.append(file_path)

    # Parse channel_distances and impact_times
    channel_distances_list = [float(d.strip()) for d in channel_distances.split(',')]
    impact_times_list = [float(t.strip()) for t in impact_times.split(',')]

    if len(channel_distances_list) != len(other_csv_paths):
        return templates.TemplateResponse("results.html", {"request": request, "error": "Mismatch between number of channel distances and uploaded CSVs."})

    # Process reference CSV (force measurements)
    ref_df = read_csv_with_metadata(ref_csv_path)
    exact_impact_times = get_exact_impact_times(ref_df, impact_times_list)
    force_measurements = preprocess_data(ref_df, exact_impact_times)

    ltm_measurements = {}

    # Process other CSVs (vibration measurements) and calculate transfer mobility
    for distance, other_csv_path in zip(channel_distances_list, other_csv_paths):
        other_df = read_csv_with_metadata(other_csv_path)
        vibration_measurements = preprocess_data(other_df.iloc[1:, :], exact_impact_times)

        # Perform subtraction (Vibration Level - Force Level)
        current_distance_measurements = {}
        for band_center, vib_level in vibration_measurements.items():
            force_level = force_measurements.get(band_center, np.nan)
            if not pd.isna(vib_level) and not pd.isna(force_level):
                current_distance_measurements[band_center] = vib_level - force_level
            else:
                current_distance_measurements[band_center] = np.nan # Or handle as appropriate
        
        ltm_measurements[distance] = current_distance_measurements

    # Initialize LineTransferMobility
    ltm = LineTransferMobility(
        measurements=ltm_measurements,
        train_length=train_length,
        receiver_offset=receiver_distance,
        source_depth=source_depth
    )

    # Perform calculations and generate plots
    ltm.regress_point_sources()
    ltm.compute_all_line_responses()

    # Generate Plotly figures using LTM class methods
    plots = {}
    # plots["point_regressions"] = ltm.plot_point_regressions().to_html(full_html=False, include_plotlyjs=False)
    plots["measurements_level_vs_distance"] = ltm.plot_measurements_level_vs_distance().to_html(full_html=False, include_plotlyjs=False)
    plots["measurements_level_vs_frequency"] = ltm.plot_measurements_level_vs_frequency().to_html(full_html=False, include_plotlyjs=False)
    plots["line_responses"] = ltm.plot_line_responses().to_html(full_html=False, include_plotlyjs=False)

    # Clean up temporary files
    os.remove(ref_csv_path)
    for other_csv_path in other_csv_paths:
        os.remove(other_csv_path)
    os.rmdir(temp_dir)
    subtitle = f"train_length = {train_length}m, receiver_offset = {receiver_distance}m, source_depth = {source_depth}m"
    subtitle2 = f"impact times = {[int(t) for t in exact_impact_times]}"
    return templates.TemplateResponse("results.html", {"request": request, "plots": plots, "subtitle": subtitle, "subtitle2": subtitle2})

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
                <label for="reference_csv">Reference CSV:</label><br>
                <input type="file" id="reference_csv" name="reference_csv"><br><br>
                
                <label for="other_csvs">Other CSV Files (can upload multiple):</label><br>
                <input type="file" id="other_csvs" name="other_csvs" multiple><br><br>
                
                <label for="channel_distances">Channel Distances (comma-separated):</label><br>
                <input type="text" id="channel_distances" name="channel_distances"><br><br>
                
                <label for="impact_times">Impact Times (comma-separated):</label><br>
                <input type="text" id="impact_times" name="impact_times"><br><br>
                
                <label for="train_length">Train Length (m):</label><br>
                <input type="number" id="train_length" name="train_length" step="any"><br><br>
                
                <label for="receiver_distance">Receiver Distance (m):</label><br>
                <input type="number" id="receiver_distance" name="receiver_distance" step="any"><br><br>
                
                <label for="source_depth">Source Depth (m):</label><br>
                <input type="number" id="source_depth" name="source_depth" step="any" value="0.0"><br><br>
                
                <input type="submit" value="Analyze">
            </form>
        </body>
        </html>
        """)

    uvicorn.run(app, host="0.0.0.0", port=8000)
