import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/imu.csv')

# Initialize variables to keep track of the adjusted timestamp and the previous timestamp
adjusted_timestamp = df['timestamp'][0]
previous_timestamp = df['timestamp'][0]

# Create a list to store the adjusted timestamps
adjusted_timestamps = []

# Iterate through the timestamps and adjust as needed
# The adjusted timestamp should not simply add itself to the previous, it should 
# add its difference between the two and the first timestamp where the reset occurred. 
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

ax1.plot(df['adjusted_timestamp'], df['accel_x'], label='X Axis', color='r')
ax1.set_ylabel('Acceleration m/s^2')
ax1.legend()

ax2.plot(df['adjusted_timestamp'], df['accel_y'], label='Y Axis', color='g')
ax2.set_ylabel('Acceleration m/s^2')
ax2.legend()

ax3.plot(df['adjusted_timestamp'], df['accel_z'], label='Z Axis', color='b')
ax3.set_xlabel('Timestamp Microseconds')
ax3.set_ylabel('Acceleration m/s^2')

plt.suptitle('Acceleration Data Over Time')
plt.show()