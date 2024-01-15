import pandas as pd

def normalize_timestamps(file):
    df = pd.read_csv(file)

    diff = [0]
    for i in range(1, len(df)):
        if df.iloc[i - 1, -1] < df.iloc[i, -1]:
            diff.append(df.iloc[i, -1] - df.iloc[i - 1, -1])
        else:
            diff.append(diff[-1])

    acc = df.iloc[0, -1]
    for i in range(1, len(df)):
        acc += diff[i]
        df.iloc[i, -1] = acc
    
    df.to_csv(f"new_{file}", index=False)

files = ['data/air.csv', 'data/gps1.csv', 'data/imu.csv', 'data/nav.csv', 'data/quat.csv', 'data/utc.csv', 'data/vel.csv']

for f in files:
    normalize_timestamps(f)