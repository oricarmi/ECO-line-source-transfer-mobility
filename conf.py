THIRD_OCTAVE_BANDS = {'4 Hz': 4, '5 Hz': 5, '6.3 Hz': 6.3, '8 Hz': 8, '10 Hz': 10, '12.5 Hz': 12.5, '16 Hz': 16, '20 Hz': 20, '25 Hz': 25, '31.5 Hz': 31.5, '40 Hz': 40, '50 Hz': 50, '63 Hz': 63, '80 Hz': 80, '100 Hz': 100, '125 Hz': 125, '160 Hz': 160, '200 Hz': 200, '250 Hz': 250, '315 Hz': 315, '400 Hz': 400}
IMPACT_SEARCH_RANGE = 8 # seconds
V_Ref = 5e-8 # 50 nanometer
F_Ref = 1
REGRESSION_LOG_MULTIPLIER = 20
y_axis_titles_pstm = {
    "imperial": "dB rel μin/s/ lbf",
    "metric": "dB rel (5e-8 m/sec) / N"
}
y_axis_titles_lstm = {
    "imperial": "dB rel (μin/s) / (lbf/√in)",
    "metric": "dB rel (5e-8 m/sec) / (N/√m)"
}

metric_to_empirial_pstm = 18.85 # dB (add this to the metric to get to imperial)
metric_to_empirial_lstm = 34.8 # dB (add this to the metric to get to imperial)