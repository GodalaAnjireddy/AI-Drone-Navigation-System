import airsim
import time
import math
import matplotlib.pyplot as plt

class SmartDrone:

    def __init__(self):

        print("Connecting to AirSim...")

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.takeoffAsync().join()

        print("Drone Ready 🚁")

        # 🔽 SMALL DISTANCE PATH
        self.waypoints = [
            [10, 0, -8],
            [10, 10, -8],
            [0, 10, -8],
            [-10, 10, -8],
            [-10, 0, -8],
            [0, -10, -8],
            [10, -10, -8],
            [0, 0, -8]
        ]

        self.current = 0

        # 🗺 Map
        self.path_x = []
        self.path_y = []

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.line, = self.ax.plot([], [], 'b-', linewidth=2)
        self.point, = self.ax.plot([], [], 'ro')

        self.ax.set_title("Drone Path (GPS + IMU + Battery)")
        self.ax.grid(True)

        # 🔋 BATTERY SYSTEM
        self.battery = 100
        self.low_battery_threshold = 25
        self.home = [0, 0, -8]

        # 🚀 Speed
        self.speed = 5

    # 📡 GPS + IMU
    def get_sensor_data(self):

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position

        imu = self.client.getImuData()
        ang_vel = imu.angular_velocity

        print(f"📍 GPS: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
        print(f"🧭 IMU: ({ang_vel.x_val:.2f}, {ang_vel.y_val:.2f}, {ang_vel.z_val:.2f})")

        return pos.x_val, pos.y_val, pos.z_val

    # 🔋 Battery update
    def update_battery(self):

        self.battery -= 2

        # adjust speed based on battery
        if self.battery > 60:
            self.speed = 6
        elif self.battery > 30:
            self.speed = 5
        else:
            self.speed = 3

        print(f"🔋 Battery: {self.battery}% | Speed: {self.speed} m/s")

    # 🔌 Return and charge
    def return_and_charge(self):

        print("\n⚠ Low Battery! Returning Home...")

        self.client.moveToPositionAsync(
            self.home[0], self.home[1], self.home[2], 5
        ).join()

        print("🔌 Charging...")
        time.sleep(3)

        self.battery = 100
        print("✅ Fully Charged!")

    # 🚧 Collision Avoidance
    def avoid_obstacle(self):

        front = self.client.getDistanceSensorData("DistanceFront").distance
        left = self.client.getDistanceSensorData("DistanceLeft").distance
        right = self.client.getDistanceSensorData("DistanceRight").distance

        if front < 5:

            print("⚠ Obstacle detected!")

            if left > right:
                self.client.moveByVelocityAsync(0, -3, 0, 1).join()
            else:
                self.client.moveByVelocityAsync(0, 3, 0, 1).join()

            return True

        return False

    # 🗺 Map
    def update_map(self, x, y):

        self.path_x.append(x)
        self.path_y.append(y)

        self.line.set_data(self.path_x, self.path_y)
        self.point.set_data([x], [y])

        self.ax.relim()
        self.ax.autoscale_view()

        plt.pause(0.1)

    def move_to_waypoint(self):

        # 🔋 Check battery
        if self.battery <= self.low_battery_threshold:
            self.return_and_charge()

        target = self.waypoints[self.current]

        print(f"\n➡ Moving to {target}")

        while True:

            if self.avoid_obstacle():
                continue

            self.client.moveToPositionAsync(
                target[0], target[1], target[2], self.speed
            ).join()

            x, y, z = self.get_sensor_data()

            self.update_map(x, y)
            self.update_battery()

            dist = math.sqrt(
                (target[0] - x)**2 +
                (target[1] - y)**2 +
                (target[2] - z)**2
            )

            if dist < 2:
                print("✅ Waypoint reached")
                break

        self.current += 1

        if self.current >= len(self.waypoints):
            return True

        return False

    def run(self):

        complete = False

        while not complete:
            complete = self.move_to_waypoint()
            time.sleep(1)

        print("\n🎯 MISSION COMPLETE")

        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    drone = SmartDrone()
    drone.run()