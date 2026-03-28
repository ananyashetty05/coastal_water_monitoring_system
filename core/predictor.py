import pandas as pd

def predict(df, location):
    d = df[df["location"] == location].sort_values("date")

    last = d.iloc[-1]

    future_dates = pd.date_range(start=last["date"], periods=7)

    predictions = pd.DataFrame({
        "date": future_dates,
        "do": [last["do"]] * 7,
        "ph": [last["ph"]] * 7,
        "sulphur": [last["sulphur"]] * 7,
        "temp": [last["temp"]] * 7,
        "turbidity": [last["turbidity"]] * 7
    })

    return predictions