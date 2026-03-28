import pandas as pd
import numpy as np

def parse_csv(file):
    return pd.read_csv(file)

def generate_sample_data():
    data = {
        "location": ["A", "A", "B", "B"],
        "date": pd.date_range(start="2024-01-01", periods=4),
        "do": np.random.uniform(4, 10, 4),
        "ph": np.random.uniform(6.5, 8.5, 4),
        "sulphur": np.random.uniform(1, 5, 4),
        "temp": np.random.uniform(20, 30, 4),
        "turbidity": np.random.uniform(1, 10, 4),
        "lat": [18.5, 18.5, 19.0, 19.0],
        "lon": [73.8, 73.8, 72.8, 72.8]
    }
    return pd.DataFrame(data)

def get_stats(df, location):
    d = df[df["location"] == location]
    return {
        "do": d["do"].mean(),
        "ph": d["ph"].mean(),
        "sulphur": d["sulphur"].mean(),
        "temp": d["temp"].mean(),
        "turbidity": d["turbidity"].mean()
    }

def get_location_summaries(df):
    return df.groupby("location")[["lat", "lon"]].mean().reset_index()