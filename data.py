import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

# Read CSV Files
df_imu = pd.read_csv('new_data/imu.csv')
df_air = pd.read_csv('new_data/air.csv')
df_vel = pd.read_csv('new_data/vel.csv')
df_gps1 = pd.read_csv('new_data/gps1.csv')

# Select Specific Columns
df_imu = df_imu[["accel_x", "accel_y", "accel_z", "temp", "timestamp"]]
df_air = df_air[["pressure_abs", "alt", "timestamp"]]
df_vel = df_vel[["vel_down", "timestamp"]]
df_gps = df_gps1[["latitude", "longitude", "timestamp"]]

# df_imu['timestamp_datetime'] = pd.to_datetime(df_imu['timestamp'])
# timestamps = df_imu['timestamp_datetime']
# timestamps_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds()
# df_imu.sort_values('timestamp', inplace=True)
df_imu['timestamp'] = df_imu['timestamp'].astype(int)
df_air['timestamp'] = df_air['timestamp'].astype(int)
df_vel['timestamp'] = df_vel['timestamp'].astype(int)
# df_gps['timestamp'] = df_gps['timestamp'].astype(int)

df_vel = pd.concat([df_vel,df_gps], axis=0)


df_imu.set_index('timestamp', inplace=True)
df_air.set_index('timestamp', inplace=True)
df_vel.set_index('timestamp', inplace=True)
# df_gps.set_index('timestamp', inplace=True)
df_imu = df_imu.iloc[df_imu.index.get_loc(2880800000):]

df_vel = df_vel.iloc[df_vel.index.get_loc(2879800108):]

# velocity_z = []
vel_z = 0.0
vel=0.0
times = []
# for i in range(1, len(df_imu)):
#     time_diff = (df_air['timestamp'][i] - df_air['timestamp'][i-1]) /1000000
#     alt_diff = df_air["alt"][i] - df_air["alt"][i-1]
#     vel_z = alt_diff / time_diff if time_diff != 0 else 0
#     velocity_z.append(vel_z)

gravity = -9.81  # gravity in m/s^2

velocity_z = [0]  # start with velocity 0
for i in range(1, len(df_imu)):
    time_diff = (df_imu.index[i]  - df_imu.index[i-1]) / 1000000
    accel_z_without_gravity = df_imu.loc[df_imu.index[i], "accel_z"]
    velocity_z.append(velocity_z[i-1] + accel_z_without_gravity * time_diff)

# from pykalman import KalmanFilter
# import numpy as np

# # Initialize the Kalman filter
# kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

# # Define the state transition matrix and control-input model
# F = np.array([[1, 1], [0, 1]])  # state transition matrix
# B = np.array([[0.5], [1]])  # control-input model

# # Define the process and measurement noise covariances
# Q = np.eye(2)  # process noise
# R = 1  # measurement noise

# # Initialize the state and covariance
# state = np.array([0, 0])  # initial state (velocity and acceleration)
# cov = np.eye(2)  # initial covariance
# df_imu['timestamp'] = df_imu['timestamp'].astype(int)
# df_air['timestamp'] = df_air['timestamp'].astype(int)

# df_imu.set_index('timestamp', inplace=True)
# df_air.set_index('timestamp', inplace=True)
# print(df_imu.index.get_loc(2880800000))
# df_imu = df_imu.iloc[df_imu.index.get_loc(2880800000):]
# print(len(df_imu))
# velocity_z = [state[0]]
# for i in range(1, len(df_imu)):
#     # Calculate the time difference
#     time_diff = (df_imu.index[i] - df_imu.index[i-1]) / 1000000

#     # Update the state transition matrix and control-input model
#     F[0, 1] = time_diff
#     B[0, 0] = longitude0.5 * time_diff**2

#     # Predict the next state and covariance
#     state = np.dot(F, state) + np.dot(B, df_imu.loc[df_imu.index[i], "accel_z"])
#     cov = np.dot(F, np.dot(cov, F.T)) + Q

#     # Update the state and covariance with the measurement
#     y = df_imu.loc[df_imu.index[i], "accel_z"] - state[1]  # measurement residual
#     S = np.dot(F, np.dot(cov, F.T)) + R  # residual covariance
#     K = np.dot(cov, np.linalg.inv(S))  # Kalman gain
#     state = state + np.dot(K, y)
#     cov = cov - np.dot(K, np.dot(S, K.T))

#     # Append the velocity to the list
#     velocity_z.append(state[0])
# temp = []
# # velocity_z = velocity_z.append(np.array([0,0]))
# for i in range(1, len(velocity_z)):
#     # velocity_z[i] = velocity_z[i] + velocity_z[i-1]
#     # print(velocity_z[i][0])
#     temp.append(velocity_z[i][0])
# velocity_z = [vel[0] for vel in velocity_z]
# temp.append(0)
# velocity_z.append(0)


# @####
# df_imu['vel_down'] = np.array(velocity_z) 
# df_imu['vel_down'] = -cumtrapz(df_imu['accel_z'] , initial=0) 

# Convert Timestamps to Integers because we want to use it as an index. 


# Cut off the first 1000 entries 

combined_df = pd.concat([df_imu, df_air, df_vel], axis=0)
# combined_df = pd.concat([df_imu, df_air, df_vel], ignore_index=True)
# Bunch of garbage at the start. Cut it off.
# combined_df = combined_df.iloc[1442770471:]

# combined_df.fillna('', inplace=True)

combined_df.interpolate()

if (combined_df.index.is_monotonic_increasing):
    print("IMU Timestamps are in order")

combined_df.to_csv('combined_data.csv', index=True)

# # Sort by Timestamp
# combined_df = combined_df.sort_values(by='timestamp', ascending=False)

# combined_df = combined_df.groupby('timestamp').agg(lambda x: ','.join(x.astype(str))).reset_index()
# combined_df.to_csv('combined_data.csv', index=False)


plt.figure(figsize=(10, 6))
plt.plot(combined_df['vel_down'], label='Velocity Down (z)')
plt.plot(combined_df['alt'], label='Altitude', color='red')  # Add this line
plt.plot(combined_df['longitude'], label='Long', color='green')
plt.xlabel('Time')
plt.ylabel('Velocity Down (z)')
plt.title('Rocket Velocity Down Including Initial Velocity')
plt.legend()
plt.grid(True)
plt.show()