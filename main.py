import airsim
import time
import cv2
import numpy as np
import math

# -----------------------------
# A* PATH PLANNING FUNCTION
# -----------------------------

def astar(grid, start, goal):

    open_list = []
    open_list.append(start)

    came_from = {}

    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    f_score = {start: heuristic(start, goal)}

    while open_list:

        current = min(open_list, key=lambda x: f_score.get(x, float('inf')))

        if current == goal:

            path = []

            while current in came_from:
                path.append(current)
                current = came_from[current]

            path.append(start)
            path.reverse()

            return path

        open_list.remove(current)

        neighbors = [
            (current[0]+1,current[1]),
            (current[0]-1,current[1]),
            (current[0],current[1]+1),
            (current[0],current[1]-1)
        ]

        for neighbor in neighbors:

            x,y = neighbor

            if x<0 or y<0 or x>=len(grid) or y>=len(grid[0]):
                continue

            if grid[x][y] == 1:
                continue

            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor,goal)

                if neighbor not in open_list:
                    open_list.append(neighbor)

    return []

# -----------------------------
# GRID MAP
# -----------------------------

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

# Calculate path
path = astar(grid,start,goal)

print("Calculated Path:", path)

# Save path for visualization
with open("path.txt","w") as f:
    for p in path:
        f.write(str(p[0])+","+str(p[1])+"\n")

# -----------------------------
# CONNECT TO AIRSIM
# -----------------------------

client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()

# -----------------------------
# TOP-DOWN CAMERA SETUP
# -----------------------------

pitch_rad = math.radians(-90)
yaw_rad = 0

position = airsim.Vector3r(0,0,0)
orientation = airsim.to_quaternion(pitch_rad,0,yaw_rad)

camera_pose = airsim.Pose(position,orientation)

client.simSetCameraPose("0",camera_pose)

# -----------------------------
# CAPTURE TERRAIN IMAGES
# -----------------------------

heights = [5,10,15,20,25,30]

for h in heights:

    print("Capturing terrain at",h,"meters")

    client.moveToPositionAsync(0,0,-h,5).join()

    time.sleep(1)

    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])

    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)

    img_rgb = img1d.reshape(responses[0].height,responses[0].width,3)

    filename = "terrain_"+str(h)+"m.png"

    cv2.imwrite(filename,img_rgb)

# -----------------------------
# DRONE NAVIGATION
# -----------------------------

altitude = -5

for point in path:

    x = point[0] * 30
    y = point[1] * 30

    print("Moving to:",point)

    # Distance sensors
    front = client.getDistanceSensorData("DistanceFront").distance
    left = client.getDistanceSensorData("DistanceLeft").distance
    right = client.getDistanceSensorData("DistanceRight").distance

    if front < 5:

        print("Obstacle detected!")

        if left > right:
            client.moveByVelocityAsync(0,-3,0,2).join()
        else:
            client.moveByVelocityAsync(0,3,0,2).join()

    client.moveToPositionAsync(x,y,altitude,10).join()

    time.sleep(1)

    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])

    img1d = np.frombuffer(responses[0].image_data_uint8,dtype=np.uint8)

    img_rgb = img1d.reshape(responses[0].height,responses[0].width,3)

    filename = "capture_"+str(point[0])+"_"+str(point[1])+".png"

    cv2.imwrite(filename,img_rgb)

print("Destination reached")

# -----------------------------
# LANDING
# -----------------------------

print("Landing...")

client.landAsync().join()

client.armDisarm(False)
client.enableApiControl(False)

print("Mission completed")