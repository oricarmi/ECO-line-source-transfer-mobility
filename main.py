from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import pandas as pd
from preprocess_data import get_exact_impact_times, preprocess_data
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

    # Process reference CSV
    ref_df = pd.read_csv(ref_csv_path)
    exact_impact_times = get_exact_impact_times(ref_df.iloc[1:, :], impact_times_list)

    if len(channel_distances_list) != len(other_csv_paths):
        return templates.TemplateResponse("results.html", {"request": request, "error": "Mismatch between number of channel distances and uploaded CSVs."})


    # Process other channels (not ref)
    ltm_measurements = {}
    for i, other_csv_path in enumerate(other_csv_paths):
        df = pd.read_csv(other_csv_path)
        ltm_measurements[channel_distances_list[i]] = preprocess_data(df, ref_df, exact_impact_times)

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

    # Generate Plotly figures
    plots = {}

    # Plot 1: Point-source regressions
    fig_point_regressions = go.Figure()
    for fc, reg in ltm.point_regressions.items():
        ds = np.logspace(np.log10(min(ltm.distances)), np.log10(max(ltm.distances)), 200)
        fig_point_regressions.add_trace(go.Scatter(x=ds, y=reg["predict"](ds), mode='lines', name=f"{fc:.1f} Hz"))
    fig_point_regressions.update_layout(
        title="Point-source regressions",
        xaxis_title="Log Distance (m)",
        yaxis_title="Level (dB)",
        xaxis_type="log"
    )
    plots["point_regressions"] = fig_point_regressions.to_html(full_html=False, include_plotlyjs='cdn')

    # Plot 2: Measurements - Level vs Distance
    fig_measurements_distance = go.Figure()
    for band_center in ltm.band_centers:
        levels = [ltm.measurements[d][band_center] for d in ltm.distances]
        fig_measurements_distance.add_trace(go.Scatter(x=ltm.distances, y=levels, mode='markers+lines', name=f"{band_center:.1f} Hz"))
    fig_measurements_distance.update_layout(
        title="Measurements - Level vs Distance",
        xaxis_title="Distance (m)",
        yaxis_title="Level (dB)",
    )
    plots["measurements_level_vs_distance"] = fig_measurements_distance.to_html(full_html=False, include_plotlyjs='cdn')

    # Plot 3: Measurements - Level vs Frequency
    fig_measurements_frequency = go.Figure()
    for d in ltm.distances:
        levels = [ltm.measurements[d][fc] for fc in ltm.band_centers]
        fig_measurements_frequency.add_trace(go.Scatter(x=np.log10(ltm.band_centers), y=levels, mode='markers+lines', name=f"{d} m"))
    fig_measurements_frequency.update_layout(
        title="Measurements - Level vs Frequency",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Level (dB)",
        xaxis=dict(
            tickvals=np.log10(ltm.band_centers),
            ticktext=ltm.band_centers,
        )
    )
    plots["measurements_level_vs_frequency"] = fig_measurements_frequency.to_html(full_html=False, include_plotlyjs='cdn')

    # Plot 4: Line responses
    fig_line_responses = go.Figure()
    fig_line_responses.add_trace(go.Scatter(x=np.log10(ltm.band_centers), y=list(ltm.line_responses.values()), mode='markers+lines'))
    fig_line_responses.update_layout(
        title=f"Line response - Receiver Distance {receiver_distance}m. Train Length {train_length}m",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Level (dB)",
        xaxis=dict(
            tickvals=np.log10(ltm.band_centers),
            ticktext=ltm.band_centers,
        )
    )
    plots["line_responses"] = fig_line_responses.to_html(full_html=False, include_plotlyjs='cdn')

    # Clean up temporary files
    os.remove(ref_csv_path)
    for other_csv_path in other_csv_paths:
        os.remove(other_csv_path)
    os.rmdir(temp_dir)

    return templates.TemplateResponse("results.html", {"request": request, "plots": plots})

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
