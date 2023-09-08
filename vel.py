import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/nav.csv')
df_sort = df.sort_values(by='vel_down', ascending=True)
print(df_sort);

# Initialize variables to keep track of the adjusted timestamp and the previous timestamp
adjusted_timestamp = df['timestamp'][0]
previous_timestamp = df['timestamp'][0]

# Create a list to store the adjusted timestamps
adjusted_timestamps = []

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

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,6))

ax1.plot(df['adjusted_timestamp'], df['vel_north'], label='X Axis', color='r')
ax1.set_ylabel('Velocity m/s')
ax1.legend()

ax2.plot(df['adjusted_timestamp'], df['vel_east'], label='Y Axis', color='g')
ax2.set_ylabel('Velocity m/s')
ax2.legend()

ax3.plot(df['adjusted_timestamp'], df['vel_down'], label='Z Axis', color='b')
ax3.set_xlabel('Timestamp Microseconds')
ax3.set_ylabel('Velocity m/s')

plt.suptitle('Velocity Data Over Time')
plt.show()