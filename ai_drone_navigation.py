import airsim
import time
import cv2
import numpy as np
from path_planner import astar

# Grid map (0 = free space, 1 = obstacle)
grid = [
[0,0,0,0,0,0,0],
[0,1,1,0,0,1,0],
[0,0,0,0,0,0,0],
[0,1,0,1,0,0,0],
[0,0,0,0,0,1,0],
[0,0,1,0,0,0,0],
[0,0,0,0,0,0,0]
]

start = (0,0)
goal = (6,6)

# Get path using A*
path = astar(grid,start,goal)

print("Calculated Path:", path)

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()

altitude = -5

for point in path:

    x = point[0] * 30
    y = point[1] * 30

    print("Moving to waypoint:", point)

    # Distance sensors
    front = client.getDistanceSensorData("DistanceFront").distance
    left = client.getDistanceSensorData("DistanceLeft").distance
    right = client.getDistanceSensorData("DistanceRight").distance

    if front < 5:

        print("Obstacle detected!")

        if left > right:
            print("Avoiding left")
            client.moveByVelocityAsync(0, -3, 0, 2).join()

        else:
            print("Avoiding right")
            client.moveByVelocityAsync(0, 3, 0, 2).join()

    # Move drone
    client.moveToPositionAsync(x, y, altitude, 10).join()

    time.sleep(1)

    # Capture camera image
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])

    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)

    img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

    filename = "capture_" + str(point[0]) + "_" + str(point[1]) + ".png"

    cv2.imwrite(filename, img_rgb)

    print("Image captured:", filename)

print("Destination reached!")

print("Landing...")
client.landAsync().join()

client.armDisarm(False)
client.enableApiControl(False)