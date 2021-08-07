import cv2
import numpy as np
from pose import *
from representation import *
from objects import *
import utils.camera_streamer as cs
import threading

DEFAULT_CAMERA_POSITION = (-1, -1, 1)
DEFAULT_CAMERA_TARGET = (0., 0., 0.)
DEFAULT_WORLD_CAM_POSE = get_cam_pose(DEFAULT_CAMERA_POSITION, DEFAULT_CAMERA_TARGET)
DEFAULT_CAMERA_DIMS = (720, 1280, 3)
DEFAULT_K = cs.mac_K
DEFAULT_D = 0


class World:

    def __init__(self, name, bodies, cameras, camera_pose=DEFAULT_WORLD_CAM_POSE, camera_dims=DEFAULT_CAMERA_DIMS, K=DEFAULT_K, D=DEFAULT_D):
        self.bodies = bodies
        self.cameras = cameras
        self.name = name
        self.camera_dims = camera_dims
        self.K = K
        self.D = D
        self.camera_pose = camera_pose
        cv2.namedWindow(name)

    def draw(self):
        frame = np.zeros(self.camera_dims)
        for body in self.bodies:
            pose = Pose(0,0,0,0,0.2,1) # body.pose @ self.camera_pose.invert()
            body.representation.draw(frame, self.K, self.D, pose)

        for camera in self.cameras:
            pose = Pose(-0.4, 0.5, 0, -0.2, -0.2, 1) # camera.pose @ self.camera_pose.invert()
            camera.representation.draw(frame, self.K, self.D, pose)
        return frame


if __name__ == '__main__':
    bodies = []
    cameras = []

    b0 = RetinaBody("World", {1: (-1, 1, 0), 2: (1, 1, 0), 3: (1, -1, 0), 4: (-1, -1, 0)}, ((0, 0, 0), (0, 0, 0)))
    b1 = RetinaBody("Link", {1: (-0.1, 0.1, 0)}, ((0, 0, 0.), (0, 0, 0.2)), AxisArrow(axis=0))

    bodies.append(b0)
    bodies.append(b1)

    c0 = RetinaCamera(cs.WebcamStreamer(0, cs.mac_K, 0))

    cameras.append(c0)

    world = World("World", bodies, cameras)

    while True:
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            pass

        cv2.imshow(world.name, world.draw())

    for a in range(5):
        cv2.destroyWindow(world.name)
