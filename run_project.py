import airsim
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# PATH
path = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,4),(2,4),(3,4),(4,4)]

# MAP SETUP
plt.ion()

fig, ax = plt.subplots()

x_vals = [p[0] for p in path]
y_vals = [p[1] for p in path]

ax.plot(x_vals, y_vals, 'b-')

drone_plot, = ax.plot([], [], 'ro', markersize=10)

ax.scatter(x_vals[0], y_vals[0], color="green", label="Start")
ax.scatter(x_vals[-1], y_vals[-1], color="red", label="Destination")

ax.set_title("Live Drone Navigation Map")
ax.grid(True)
plt.legend()

# CONNECT AIRSIM
client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()

altitude = -5

# DRONE LOOP
for point in path:

    x = point[0] * 30
    y = point[1] * 30

    print("Moving to:", point)

    # CHECK OBSTACLE
    front = client.getDistanceSensorData("DistanceFront").distance
    left = client.getDistanceSensorData("DistanceLeft").distance
    right = client.getDistanceSensorData("DistanceRight").distance

    if front < 5:

        print("Obstacle detected!")

        if left > right:

            print("Moving Left")
            client.moveByVelocityAsync(0, -3, 0, 2).join()

        else:

            print("Moving Right")
            client.moveByVelocityAsync(0, 3, 0, 2).join()

    client.moveToPositionAsync(x, y, altitude, 10).join()

    # UPDATE MAP
    drone_plot.set_data([point[0]], [point[1]])
    plt.pause(2)

    # CAMERA CAPTURE
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])

    if responses and responses[0].image_data_uint8:

        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = cv2.imdecode(img1d, cv2.IMREAD_COLOR)

        if img is not None:

            filename = "capture_" + str(point[0]) + "_" + str(point[1]) + ".png"
            cv2.imwrite(filename, img)

            print("Captured:", filename)

print("Landing...")

client.landAsync().join()

client.armDisarm(False)
client.enableApiControl(False)

plt.ioff()
plt.show()