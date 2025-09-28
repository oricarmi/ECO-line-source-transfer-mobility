from utils import get_exact_impact_times, preprocess_data, read_csv_with_metadata

if __name__ == '__main__':
    path = '/Users/oriccarmi/Documents/ECO/railroad_lstm/data/field_experiment_herzliya/11121_TEST H_021_3 Octave1 CH1.csv'
    df = read_csv_with_metadata(path)
    a = 5