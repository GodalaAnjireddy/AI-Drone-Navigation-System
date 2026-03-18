import cv2
import glob

def create_terrain_map():

    files = sorted(glob.glob("capture_*.png"))

    images = []

    for file in files:

        img = cv2.imread(file)

        if img is not None:
            images.append(img)

    print("Images Loaded:",len(images))

    if len(images) < 2:

        print("Not enough images for terrain mapping")
        return

    stitcher = cv2.Stitcher_create()

    status,stitched = stitcher.stitch(images)

    if status == 0:

        cv2.imwrite("terrain_map.png",stitched)

        print("Terrain Map Created: terrain_map.png")

        cv2.imshow("Terrain Map",stitched)

        cv2.waitKey(0)

    else:

        print("Image stitching failed")