import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/nav.csv')

# Initialize variables to keep track of the adjusted timestamp and the previous timestamp
adjusted_timestamp = 0
previous_timestamp = df['timestamp'][0]

# Create a list to store the adjusted timestamps
adjusted_timestamps = []

# Iterate through the timestamps and adjust as needed
for timestamp in df['timestamp']:
    if timestamp < previous_timestamp:
        adjusted_timestamp += 1/40
    else:
        # Add the current timestamp to the adjusted timestamp
        adjusted_timestamp += 1/40
    adjusted_timestamps.append(adjusted_timestamp)
    previous_timestamp = timestamp

# Add the adjusted timestamps to the DataFrame
df['adjusted_timestamp'] = adjusted_timestamps

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,6))

ax1.plot(df['adjusted_timestamp'], df['latitude'], label='Lat', color='r')
ax1.set_ylabel('Lat')
ax1.legend()

ax2.plot(df['adjusted_timestamp'], df['longitude'], label='Lon', color='g')
ax2.set_ylabel('Long')
ax2.legend()

ax3.plot(df['adjusted_timestamp'], df['alt'], label='Alt', color='b')
ax3.set_xlabel('Timestamp Seconds')
ax3.set_ylabel('Alt')

plt.suptitle('GPS Data Over Time')
plt.show()