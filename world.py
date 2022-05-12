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
DEBUG = False


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
            # print(body.pose)
            if body.pose:
                # print("DRAWING NON NONE")
                body.representation.draw(frame, self.K, self.D, self.camera_pose, body.pose)
            # else:
                # print(self.name, "not detected")

        for camera in self.cameras:
            # print(camera.pose)
            if camera.pose:
                camera.representation.draw(frame, self.K, self.D, self.camera_pose, camera.pose)

        if DEBUG:
            print("Drew {} cameras and {} bodies".format(len(self.cameras), len(self.bodies)))
        return frame

    def display(self):
        cv2.imshow(self.name, self.draw())
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            for a in range(5):
                cv2.destroyWindow(self.name)

if __name__ == '__main__':
    bodies = []
    cameras = []

    # world_point_dict = {1: (-0.5, 0.5, 0), 2: (0.5, 0.5, 0), 3: (0.5, -0.5, 0), 4: (-0.5, -0.5, 0), 5: (0, 0, 0)}
    # b0 = RetinaBody("World", world_point_dict, ((0, 0, 0), (0, 0, 0)), color=RED)
    b1 = RetinaBody("Robot Link", {}, Pose((0,0,0),(0,0,0)),color=YELLOW,representation=AxisArrow())
    # bodies.append(b0)
    bodies.append(b1)

    # c0 = RetinaCamera(cs.WebcamStreamer(0, cs.mac_K, 0), None, get_cam_pose((0.5, 0.5, 0.5), (0, 0, 0)))
    # print(c0.pose.invert())
    # cameras.append(c0)

    bodies.append(RetinaBody(representation=AxisRectangle("World", [GREEN, RED, BLUE, RED], top_left=(-0.5, 0.5, 0), width=1.0, height=1.0)))
    bodies.append(RetinaBody(
        representation=Axes()))


    world = World("World", bodies, cameras)

    while True:
        world.camera_pose = Pose(0,0,0.04,0,0,0) @ world.camera_pose
        world.bodies[0].pose = Pose(0,0,0.04,0,0,-0.001) @ world.bodies[0].pose
        world.display()
