import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

# Read CSV Files
df_imu = pd.read_csv('data/new_imu.csv')
df_air = pd.read_csv('data/new_air.csv')
df_vel = pd.read_csv('data/new_nav.csv')

# Select Specific Columns
df_imu = df_imu[["accel_x", "accel_y", "accel_z", "temp", "timestamp"]]
df_air = df_air[["pressure_abs", "alt", "timestamp"]]

# df_imu['timestamp_datetime'] = pd.to_datetime(df_imu['timestamp'])
# timestamps = df_imu['timestamp_datetime']
# timestamps_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds()
# df_imu.sort_values('timestamp', inplace=True)


velocity_z = []
vel_z = 0.0
vel=0.0
times = []
for i in range(len(df_imu)):
    time = (df_imu['timestamp'][i] - 2245000) / 1000
    vel_z += df_imu["accel_z"][i] * time
    velocity_z.append(vel_z )

df_imu['vel_down'] = -np.array(velocity_z) 
# df_imu['vel_down'] = -cumtrapz(df_imu['accel_z'] , initial=0) 

combined_df = pd.concat([df_imu, df_air], ignore_index=True)
# combined_df = pd.concat([df_imu, df_air, df_vel], ignore_index=True)

combined_df.fillna('', inplace=True)

# # Sort by Timestamp
# combined_df = combined_df.sort_values(by='timestamp', ascending=False)

# combined_df = combined_df.groupby('timestamp').agg(lambda x: ','.join(x.astype(str))).reset_index()
# combined_df.to_csv('combined_data.csv', index=False)


plt.figure(figsize=(10, 6))
plt.plot(df_imu['timestamp'], df_imu['vel_down'], label='Velocity Down (z)')
plt.xlabel('Time')
plt.ylabel('Velocity Down (z)')
plt.title('Rocket Velocity Down Including Initial Velocity')
plt.legend()
plt.grid(True)
plt.show()