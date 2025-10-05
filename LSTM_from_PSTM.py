import numpy as np
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from conf import V_Ref, F_Ref, y_axis_titles_pstm, y_axis_titles_lstm, metric_to_empirial_pstm, metric_to_empirial_lstm, REGRESSION_LOG_MULTIPLIER

class LineTransferMobility:
    def __init__(self, velocity_measurements, force_measurements, train_length, receiver_offset, velocity_measurements_iqr=None, source_depth=0.0):
        """
        velocity measurements: dict {distance (m): {band_center (Hz): level (m/s)}}
        force_measurements: dict {band center (Hz): level (N)}
        train_length: float, train length in meters
        receiver_offset: float, receiver distance from track centerline (m)
        source_depth: float, optional, vertical source depth (m)
        """
        self.force_measurements = force_measurements
        self.measurements = velocity_measurements
        self.measurements_iqr = velocity_measurements_iqr
        if self.measurements_iqr is None:
            self.measurements_iqr = {d: {fc: 0 for fc in self.band_centers} for d in self.distances}
        self.distances = sorted(velocity_measurements.keys())
        # collect all band centers from the first entry
        self.band_centers = sorted(list(next(iter(velocity_measurements.values())).keys()))
        self.train_length = train_length
        self.receiver_offset = receiver_offset
        self.source_depth = source_depth
        self.point_regressions = {}  # band -> {"a","b","predict"}
        self.lstm = {}     # band -> line response level (dB)

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
        """Perform regression for all band centers - linear decay with log of distance
        From A Prediction Procedure for Rail Transportation Groundborne Noise and Vibration
        JAMES TUMAN NELSON AND HUGH J. SAURENMAN
        Reference in FTA ref 50
        """
        for fc in self.band_centers:
            log_measurements = [REGRESSION_LOG_MULTIPLIER * np.log10(self.measurements[d][fc]) for d in self.distances]
            a, b, predict = self.regress_levels_vs_distance(self.distances, log_measurements)
            self.point_regressions[fc] = {"a": a, "b": b, "predict": predict}
        return self.point_regressions

    # --- Integration over train length ---
    def integrate_pstm_to_lstm(self, predict_level_db_fn, F, num_segments=701):
        xs = np.linspace(-self.train_length/2, self.train_length/2, num_segments)
        slant = np.sqrt(self.receiver_offset**2 + xs**2 + self.source_depth**2)        
        point_db = predict_level_db_fn(slant)
        yp_linear = 10 ** (point_db / REGRESSION_LOG_MULTIPLIER)
        # amplitude sum (according to FTA e.q B4 Appendix B)
        y_line = np.trapezoid(yp_linear, xs) / np.sqrt(self.train_length) / F # (m / s) / (N / sqrt(m))
        y_ref = V_Ref / F_Ref
        line_db = 10 * np.log10(y_line / y_ref)
        return line_db

    def compute_freq_lstm(self, band_center):
        """Compute line response for a single band."""
        if not self.point_regressions:
            self.regress_point_sources()
        reg = self.point_regressions[band_center]
        F = self.force_measurements[band_center]
        lstm_db = self.integrate_pstm_to_lstm(reg["predict"], F)
        self.lstm[band_center] = lstm_db
        return lstm_db

    def compute_lstms_all_freqs(self):
        """Compute line responses for all bands."""
        if not self.point_regressions:
            self.regress_point_sources()
        for fc in self.band_centers:
            self.compute_freq_lstm(fc)
        return self.lstm

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
    def plot_point_regressions(self, units="metric"):
        if not self.point_regressions:
            self.regress_point_sources()
        
        fig = go.Figure()
        for fc, reg in self.point_regressions.items():
            ds = np.linspace(1.0, max(self.distances)+10)
            vals = 10*np.log10(10**(reg["predict"](ds) / REGRESSION_LOG_MULTIPLIER) / self.force_measurements[fc] / V_Ref)
            if units == "imperial":
                vals += metric_to_empirial_pstm
            fig.add_trace(go.Scatter(x=ds, y=list(vals), mode='lines', name=f"{fc:.1f} Hz"))
        fig.update_layout(
            title="Point-source regressions",
            xaxis_title="Log Distance (m)",
            yaxis_title="Level (dB rel (5e-8 m/sec) / N)",
            xaxis=dict(type="log")
        )
        return fig
    
    def plot_force_measurements(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.band_centers, y=list(10*np.log10(np.array(list(self.force_measurements.values())) / F_Ref / np.sqrt(self.train_length))), mode="markers+lines"))
        fig.update_layout(
            title="Force Measurements",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Level (dB rel 1N / sqrt(m))",
            xaxis=dict(type="log"))
        return fig

    def plot_pstm_level_vs_distance(self, units="metric"):
        fig = go.Figure()
        sqrt_N = np.sqrt(len(self.distances))
        for band_center in self.band_centers:
            levels = np.array([10 * np.log10(self.measurements[d][band_center] / V_Ref / self.force_measurements[band_center]) for d in self.distances])
            errors = np.array([8.686 * self.measurements_iqr[d][band_center] / sqrt_N / V_Ref / self.force_measurements[band_center] for d in self.distances])
            errors /= levels
            if units == "imperial":
                levels += metric_to_empirial_pstm
            fig.add_trace(go.Scatter(x=self.distances, y=list(levels), error_y=dict(type='data', array=list(errors), visible=True), mode='markers+lines', name=f"{band_center:.1f} Hz"))
        fig.update_layout(
            title="Point Source Transfer Mobility - Velocity Measurements vs Log Distance",
            xaxis_title="Distance (m)",
            yaxis_title=y_axis_titles_pstm[units],
            xaxis=dict(type="log")
        )
        return fig
    
    def plot_pstm_level_vs_frequency(self, units="metric"):
        fig = go.Figure()
        sqrt_N = np.sqrt(len(self.band_centers))
        for d in self.distances:
            levels = np.array([10 * np.log10(self.measurements[d][band_center] / V_Ref / self.force_measurements[band_center]) for band_center in self.band_centers])
            errors = np.array([8.686 * self.measurements_iqr[d][band_center] / sqrt_N / V_Ref / self.force_measurements[band_center] for band_center in self.band_centers])
            errors /= levels
            if units == "imperial":
                levels += metric_to_empirial_pstm
            fig.add_trace(go.Scatter(x=self.band_centers, y=list(levels), error_y=dict(type='data', array=list(errors), visible=True), mode='markers+lines', name=f"{d} m"))
        fig.update_layout(
            title="Point Source Transfer Mobility - Velocity Measurements vs Frequency",
            xaxis_title="Frequency (Hz)",
            yaxis_title=y_axis_titles_pstm[units],
            xaxis=dict(type="log")
        )
        return fig
    
    def plot_line_responses(self, units="metric"):
        fig = go.Figure()
        levels = np.array(list(self.lstm.values()))
        if units == "imperial":
            levels += metric_to_empirial_lstm
        fig.add_trace(go.Scatter(x=self.band_centers, y=levels, mode='markers+lines', name=f"{self.receiver_offset} m"))
        fig.update_layout(
            title=f"Line Source Transfer Mobility - Receiver Distance {self.receiver_offset}m",
            xaxis_title="Frequency (Hz)",
            yaxis_title=y_axis_titles_lstm[units],
            xaxis=dict(type="log")
        )
        return fig


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
    ltm.plot_pstm_level_vs_distance()
    ltm.plot_pstm_level_vs_frequency()
    ltm.plot_point_regressions()
    # Compute regressions and line responses
    ltm.regress_point_sources()
    line_responses = ltm.compute_lstms_all_freqs()
    print("Line source responses (dB):")
    for fc, val in line_responses.items():
        print(f"  {fc:.1f} Hz: {val:.2f} dB")

    # Plot regressions
    ltm.plot_line_responses()
    
