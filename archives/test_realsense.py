# First import the library
import pyrealsense2 as rs
import numpy as np
import camera_tracking.camera_streamer as cs
import cv2

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()

print("Started Pipeline")

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        # Get frameset of color and depth

        frames = pipeline.wait_for_frames()

        print("Waited for frames")

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        print("Aligned frames")

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        print("Getting images")

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # if color_image is not None:
        #     cv2.imshow('test', cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        # print("Making colorizer")
        #
        # colorizer = rs.colorizer()
        #
        # print("Made colorizer")
        #
        # colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        #
        # print("Got colorized depth")

        cv2.imshow('test', depth_image)

        print("Showed image")

except Exception as e:
    print(e)

finally:
    pipeline.stop()
    print("Stopped pipeline")

    cv2.destroyAllWindows()
    for i in range(1, 5):
        cv2.waitKey(1)

    # streamer.close()
    exit()



