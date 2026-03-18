import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx


class LiveMap:

    def __init__(self, path):

        self.path = path

        # Convert grid path into coordinates
        x = [p[0] for p in path]
        y = [p[1] for p in path]

        geometry = [Point(xy) for xy in zip(x, y)]

        self.gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

        self.gdf = self.gdf.to_crs(epsg=3857)

        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(6,6))

        self.gdf.plot(ax=self.ax, color="blue", linewidth=2)

        ctx.add_basemap(self.ax, source=ctx.providers.OpenStreetMap.Mapnik)

        self.start = self.ax.scatter(
            self.gdf.geometry.x.iloc[0],
            self.gdf.geometry.y.iloc[0],
            color="green",
            s=120,
            label="Start"
        )

        self.goal = self.ax.scatter(
            self.gdf.geometry.x.iloc[-1],
            self.gdf.geometry.y.iloc[-1],
            color="red",
            s=120,
            label="Destination"
        )

        self.drone, = self.ax.plot([], [], 'bo', markersize=10)

        self.ax.set_title("Real World Drone Navigation Map")

        plt.legend()


    def update(self, point):

        x, y = point

        g = gpd.GeoDataFrame(
            geometry=[Point(x, y)],
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        self.drone.set_data(g.geometry.x, g.geometry.y)

        plt.pause(0.5)