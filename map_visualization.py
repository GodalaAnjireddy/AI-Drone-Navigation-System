import matplotlib.pyplot as plt
import time

# Path from A* algorithm
path = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,4),(2,4),(3,4),(4,4)]

x = [p[0] for p in path]
y = [p[1] for p in path]

plt.figure(figsize=(6,6))
plt.title("Drone Navigation Live Map")

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

plt.grid(True)

# draw full path
plt.plot(x, y, color='blue')

plt.scatter(x[0], y[0], color='green', s=100, label="Start")
plt.scatter(x[-1], y[-1], color='red', s=100, label="Destination")

drone, = plt.plot([], [], 'ro', markersize=8)

plt.legend()

for point in path:
    
    drone.set_data(point[0], point[1])
    
    plt.pause(1)

plt.show()