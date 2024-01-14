# import pandas as pd
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# # from scipy.spatial.transform import Rotation as R
# # import matplotlib.pyplot as plt

# # Read CSV Files
# df_imu = pd.read_csv('data/new_imu.csv')
# df_air = pd.read_csv('data/new_air.csv')
# df_vel = pd.read_csv('data/new_nav.csv')

# # Select Specific Columns
# df_imu = df_imu[["accel_x", "accel_y", "accel_z", "temp", "timestamp"]]
# df_air = df_air[["pressure_abs", "alt", "timestamp"]]
# df_vel = df_vel[["vel_north", "vel_east", "vel_down", "timestamp"]]


# combined_df = pd.concat([df_imu, df_air, df_vel], ignore_index=True)

# # # Replace NaN values with empty strings
# combined_df.fillna('', inplace=True)

# # # Sort by Timestamp
# combined_df = combined_df.sort_values(by='timestamp', ascending=False)

# # # Merge Rows with the Same Timestamp
# combined_df = combined_df.groupby('timestamp').agg(lambda x: ','.join(x.astype(str))).reset_index()

# # # Save to New CSV
# # combined_df.to_csv('combined_data.csv', index=False)

# # Plot Velocity (vel_down) over time
# plt.figure(figsize=(10, 6))
# plt.plot(combined_df['timestamp'], combined_df['vel_down'], label='Velocity Down (z)')
# plt.xlabel('Time')
# plt.ylabel('Velocity Down (z)')
# plt.title('Rocket Velocity Down Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV Files
df_imu = pd.read_csv('data/new_imu.csv')
df_air = pd.read_csv('data/new_air.csv')

# Select Specific Columns
df_imu = df_imu[["accel_x", "accel_y", "accel_z", "temp", "timestamp"]]
df_air = df_air[["pressure_abs", "alt", "timestamp"]]

# Merge DataFrames
combined_df = pd.merge(df_imu, df_air, on='timestamp', how='outer')

# Convert timestamp to datetime
combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])

# Calculate velocity from acceleration (using cumulative trapezoidal integration)
dt = combined_df['timestamp'].diff().dt.total_seconds()  # time difference in seconds

# Considering negative sign for upward motion
combined_df['vel_down'] = -np.cumsum(combined_df['accel_z'] * dt)

# Plot Velocity (vel_down) over time
plt.figure(figsize=(10, 6))
plt.plot(combined_df['timestamp'], combined_df['vel_down'], label='Velocity Down (z)')
plt.xlabel('Time')
plt.ylabel('Velocity Down (z)')
plt.title('Rocket Velocity Down Including Initial Velocity')
plt.legend()
plt.grid(True)
plt.show()
