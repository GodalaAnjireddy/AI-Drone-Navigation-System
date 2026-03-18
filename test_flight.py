import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()

altitude = -5

# Waypoints
path = [
(0,0),
(20,0),
(40,10),
(60,20),
(80,30),
(100,40)
]

for point in path:

    x,y = point

    print("Moving to:", point)

    front = client.getDistanceSensorData("DistanceFront").distance
    left = client.getDistanceSensorData("DistanceLeft").distance
    right = client.getDistanceSensorData("DistanceRight").distance

    if front < 5:

        print("Obstacle detected!")

        if left > right:
            print("Avoiding left")
            client.moveByVelocityAsync(0,-3,0,2).join()

        else:
            print("Avoiding right")
            client.moveByVelocityAsync(0,3,0,2).join()

    client.moveToPositionAsync(x,y,altitude,5).join()

    time.sleep(1)

print("Destination reached!")

client.landAsync().join()

client.armDisarm(False)
client.enableApiControl(False)