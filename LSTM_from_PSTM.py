"""
line_transfer_mobility_class.py

Reusable class for line transfer mobility computation
when 1/3-octave band levels are already measured.

Inputs at initialization:
  - measurements: dict of {distance_m : {band_center_Hz : level_dB}}
  - train_length: length of train (m)
  - receiver_offset: lateral receiver distance from track centerline (m)
  - source_depth: depth of source below receiver (optional, default=0.0)

Main methods:
  - regress_point_sources()
  - compute_line_response(band_center)
  - compute_all_line_responses()
  - regress_line_response_vs_distance()

"""

import numpy as np
import plotly.graph_objects as go

# hi itamar
class LineTransferMobility:
    def __init__(self, measurements, train_length, receiver_offset, source_depth=0.0):
        """
        measurements: dict {distance (m): {band_center (Hz): level_dB}}
        train_length: float, train length in meters
        receiver_offset: float, receiver distance from track centerline (m)
        source_depth: float, optional, vertical source depth (m)
        """
        self.measurements = measurements
        self.distances = sorted(measurements.keys())
        # collect all band centers from the first entry
        self.band_centers = sorted(list(next(iter(measurements.values())).keys()))
        self.train_length = train_length
        self.receiver_offset = receiver_offset
        self.source_depth = source_depth

        self.point_regressions = {}  # band -> {"a","b","predict"}
        self.line_responses = {}     # band -> line response level (dB)

    # --- Regression of point-source levels vs distance ---
    @staticmethod
    def regress_levels_vs_distance(distances, levels_db):
        distances = np.asarray(distances)
        logd = np.log10(np.maximum(distances, 1e-6))
        A = np.vstack([np.ones_like(logd), logd]).T
        coeffs, *_ = np.linalg.lstsq(A, levels_db, rcond=None)
        a, b = coeffs
        def predict(r):
            r = np.asarray(r)
            return a + b * np.log10(np.maximum(r, 1e-6))
        return a, b, predict

    def regress_point_sources(self):
        """Perform regression for all band centers."""
        for fc in self.band_centers:
            levels = [self.measurements[d][fc] for d in self.distances]
            a, b, predict = self.regress_levels_vs_distance(self.distances, levels)
            self.point_regressions[fc] = {"a": a, "b": b, "predict": predict}
        return self.point_regressions

    # --- Integration over train length ---
    def integrate_line_response(self, predict_level_db_fn, num_segments=801):
        xs = np.linspace(-self.train_length/2, self.train_length/2, num_segments)
        slant = np.sqrt(self.receiver_offset**2 + xs**2 + self.source_depth**2)
        point_db = predict_level_db_fn(slant)

        # energy sum
        linear_powers = 10 ** (point_db / 10.0)
        sum_power = np.sum(linear_powers) * (self.train_length / num_segments)
        line_db = 10 * np.log10(sum_power + np.finfo(float).eps)
        return line_db, xs, point_db

    def compute_line_response(self, band_center):
        """Compute line response for a single band."""
        if not self.point_regressions:
            self.regress_point_sources()
        reg = self.point_regressions[band_center]
        line_db, xs, point_db = self.integrate_line_response(reg["predict"])
        self.line_responses[band_center] = line_db
        return line_db

    def compute_all_line_responses(self):
        """Compute line responses for all bands."""
        if not self.point_regressions:
            self.regress_point_sources()
        for fc in self.band_centers:
            self.compute_line_response(fc)
        return self.line_responses

    # --- Regression of line response vs distance ---
    @staticmethod
    def regress_line_response_vs_distance(distances, line_levels_db, poly_order=2):
        logd = np.log10(np.maximum(distances, 1e-6))
        X = np.vstack([logd**i for i in range(poly_order+1)]).T
        coeffs, *_ = np.linalg.lstsq(X, line_levels_db, rcond=None)
        def predict(r):
            lr = np.log10(np.maximum(r, 1e-6))
            Xr = np.vstack([lr**i for i in range(poly_order+1)]).T
            return Xr.dot(coeffs)
        return coeffs, predict

    # --- Optional plotting ---
    def plot_point_regressions(self):
        if not self.point_regressions:
            self.regress_point_sources()
        
        fig = go.Figure()
        for fc, reg in self.point_regressions.items():
            ds = np.logspace(np.log10(min(self.distances)), np.log10(max(self.distances)), 200)
            fig.add_trace(go.Scatter(x=ds, y=reg["predict"](ds), mode='lines', name=f"{fc:.1f} Hz"))

        fig.update_layout(
            title="Point-source regressions",
            xaxis_title="Log Distance (m)",
            yaxis_title="Level (dB)",
            xaxis_type="log"
        )
        fig.show()
    
    def plot_measurements_level_vs_distance(self):
        fig = go.Figure()
        for band_center in self.band_centers:
            levels = [self.measurements[d][band_center] for d in self.distances]
            fig.add_trace(go.Scatter(x=self.distances, y=levels, mode='markers+lines', name=f"{band_center:.1f} Hz"))
        fig.update_layout(
            title="Measurements - Level vs Distance",
            xaxis_title="Distance (m)",
            yaxis_title="Level (dB)",
        )
        fig.show()
    def plot_measurements_level_vs_frequency(self):
        fig = go.Figure()
        for d in self.distances:
            levels = [self.measurements[d][fc] for fc in self.band_centers]
            fig.add_trace(go.Scatter(x=np.log10(self.band_centers), y=levels, mode='markers+lines', name=f"{d} m"))
        fig.update_layout(
            title="Measurements",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Level (dB)",
            xaxis=dict(
                tickvals=np.log10(self.band_centers),
                ticktext=self.band_centers,
            )
        )
        fig.show()

    def plot_line_responses(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.log10(self.band_centers), y=list(self.line_responses.values()), mode='markers+lines', name=f"{fc:.1f} Hz"))
        fig.update_layout(
            title="Line responses",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Level (dB)",
            xaxis=dict(
                tickvals=np.log10(self.band_centers),
                ticktext=self.band_centers,
            )
        )
        fig.show()


# --- Example usage ---
if __name__ == "__main__":
    # Example: measured 1/3 octave levels (synthetic)
    # measurements = {
    #     5.0:  {16.0: -20, 31.5: -25, 63.0: -30},
    #     10.0: {16.0: -23, 31.5: -29, 63.0: -35},
    #     25.0: {16.0: -28, 31.5: -33, 63.0: -40},
    #     50.0: {16.0: -32, 31.5: -37, 63.0: -44},
    #     100.0:{16.0: -36, 31.5: -41, 63.0: -48},
    # }
    measurements = { # distance: {band_center: level_dB}
        25.0: {8.0: 4, 16.0: 12, 31.5: 17, 63.0: 17, 125: 1, 170: -5},
        50.0: {8.0: -2, 16.0: 7, 31.5: 7, 63.0: 6, 125: -12, 170: -18},
        100.0: {8.0: -9, 16.0: -5, 31.5: -6, 63.0: -13, 125: -22, 170: -27},
        150.0: {8.0: -13, 16.0: -9, 31.5: -10, 63.0: -19, 125: -28, 170: -33},
        200.0: {8.0: -16, 16.0: -13, 31.5: -16, 63.0: -23, 125: -32, 170: -34},
        300.0: {8.0: -22, 16.0: -16, 31.5: -20, 63.0: -29, 125: -34, 170: -36},
}


    # Initialize class
    ltm = LineTransferMobility(measurements, train_length=70.0, receiver_offset=25.0)
    ltm.plot_measurements_level_vs_distance()
    ltm.plot_measurements_level_vs_frequency()
    ltm.plot_point_regressions()
    # Compute regressions and line responses
    ltm.regress_point_sources()
    line_responses = ltm.compute_all_line_responses()
    print("Line source responses (dB):")
    for fc, val in line_responses.items():
        print(f"  {fc:.1f} Hz: {val:.2f} dB")

    # Plot regressions
    ltm.plot_line_responses()
    
