import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/air.csv')

fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(10,6))

df_sort = df.sort_values(by='alt', ascending=False)
print(df_sort);
print(df)

# Initialize variables to keep track of the adjusted timestamp and the previous timestamp
adjusted_timestamp = df['timestamp'][0]
previous_timestamp = df['timestamp'][0]

# Create a list to store the adjusted timestamps
adjusted_timestamps = []
reset = False
# Iterate through the timestamps and adjust as needed
for timestamp in df['timestamp']:
    if timestamp < previous_timestamp:
        # If the current timestamp is less than the previous one, reset the adjustment
        adjusted_timestamp = adjusted_timestamp + timestamp
    else:
        # Add the current timestamp to the adjusted timestamp
        adjusted_timestamp += timestamp - previous_timestamp
    adjusted_timestamps.append(adjusted_timestamp)
    previous_timestamp = timestamp

# Add the adjusted timestamps to the DataFrame
df['adjusted_timestamp'] = adjusted_timestamps

df = df[df['alt'] > 0]
ax1.plot(df['adjusted_timestamp'], df['alt'], color='r')
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Alt m')

plt.suptitle('Altitude Data Over Time')
plt.show()