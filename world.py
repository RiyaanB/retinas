import cv2
import numpy as np
from pose import *
from representation import *
from objects import *
import utils.camera_streamer as cs
import threading

DEFAULT_CAMERA_POSITION = (0, -1.8, 1)
DEFAULT_CAMERA_TARGET = (0., 0., -0.1)
DEFAULT_WORLD_CAM_POSE = get_cam_pose(DEFAULT_CAMERA_POSITION, DEFAULT_CAMERA_TARGET)
DEFAULT_CAMERA_DIMS = (720, 1280, 3)
DEFAULT_K = cs.mac_K
DEFAULT_D = 0
PI = np.pi
DEBUG = True


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
            # Pose(0, 0, 0, 0, 0.2, 1)
            body.representation.draw(frame, self.K, self.D, self.camera_pose, body.pose)

        for camera in self.cameras:
            # Pose(-0.4, 0.5, 0, -0.2, -0.2, 1)
            camera.representation.draw(frame, self.K, self.D, self.camera_pose, camera.pose)

        if DEBUG:
            print("Drew {} cameras and {} bodies".format(len(self.cameras), len(self.bodies)))
        return frame


if __name__ == '__main__':
    bodies = []
    cameras = []

    # world_point_dict = {1: (-0.5, 0.5, 0), 2: (0.5, 0.5, 0), 3: (0.5, -0.5, 0), 4: (-0.5, -0.5, 0), 5: (0, 0, 0)}
    # b0 = RetinaBody("World", world_point_dict, ((0, 0, 0), (0, 0, 0)), color=RED)
    # b1 = RetinaBody("Robot Link", {}, ((0,0,0),(0,0,0)),color=YELLOW,representation=Axes())
    # bodies.append(b0)
    # bodies.append(b1)

    # c0 = RetinaCamera(cs.WebcamStreamer(0, cs.mac_K, 0), None, get_cam_pose((0.5, 0.5, 0.5), (0, 0, 0)))
    # print(c0.pose.invert())
    # cameras.append(c0)

    bodies.append(RetinaBody(representation=AxisRectangle("World", [GREEN, RED, BLUE, RED], top_left=(-0.5, 0.5, 0), width=1.0, height=1.0)))
    bodies.append(RetinaBody(
        representation=Axes()))



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
        world.camera_pose = Pose(0,0,0.01,0,0,0) @ world.camera_pose
        # b1.pose = Pose(0.1,0,0,0,0,0) @ b1.pose
        cv2.imshow(world.name, world.draw())

    for a in range(5):
        cv2.destroyWindow(world.name)
