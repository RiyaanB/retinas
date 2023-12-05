import cv2
import numpy as np
from pose import *
from representation import *
from objects import *
import utils.camera_streamer as cs
# from test_bodies.cube_body import cube0_body, cube1_body
import threading

DEFAULT_CAMERA_POSITION = (0, -1.8, 1)
DEFAULT_CAMERA_TARGET = (0., 0., -0.1)
DEFAULT_WORLD_CAM_POSE = get_cam_pose(DEFAULT_CAMERA_POSITION, DEFAULT_CAMERA_TARGET)
DEFAULT_CAMERA_DIMS = (720, 1280, 3)
DEFAULT_K = cs.mac_K
DEFAULT_D = 0
PI = np.pi
DEBUG = False


class World(threading.Thread):

    def __init__(self, name, bodies, cameras, camera_pose=DEFAULT_WORLD_CAM_POSE, camera_dims=DEFAULT_CAMERA_DIMS, K=DEFAULT_K, D=DEFAULT_D, runthread=True):
        super().__init__()
        self.bodies = bodies
        self.cameras = cameras
        self.name = name
        self.camera_dims = camera_dims
        self.K = K
        self.D = D
        self.camera_pose = camera_pose
        
        if runthread:
            cv2.namedWindow(name)
            self.__is__running__ = True
            self.start()
        else:
            self.__is__running__ = False

    def draw(self, file=None):
        frame = np.zeros(self.camera_dims)
        for body in self.bodies:
            if body.pose:
                body_file = None if file is None else []
                body.representation.draw(frame, self.K, self.D, self.camera_pose, body.pose, body_file)
                if file is not None:
                    file.append((body.name, body_file))

        for camera in self.cameras:
            if camera.pose:
                camera.representation.draw(frame, self.K, self.D, self.camera_pose, camera.pose)

        if DEBUG:
            print("Drew {} cameras and {} bodies".format(len(self.cameras), len(self.bodies)))
        return frame

    def display(self, file=None):
        disp_img = self.draw(file)
        cv2.imshow(self.name, disp_img)
        cv2.imwrite(self.name + ".png", disp_img)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            for a in range(5):
                cv2.destroyWindow(self.name)
        pass

    def run(self):
        print("RUNNNING")
        while self.__is__running__:
            # print("Drawing...")
            self.display()
            # print("Drew!")
        print("QUITTING")

if __name__ == '__main__':
    bodies = []
    cameras = []

    # world_point_dict = {1: (-0.5, 0.5, 0), 2: (0.5, 0.5, 0), 3: (0.5, -0.5, 0), 4: (-0.5, -0.5, 0), 5: (0, 0, 0)}
    # b0 = RetinaBody("World", world_point_dict, ((0, 0, 0), (0, 0, 0)), color=RED)
    # bodies.append(b0)
    # b1 = RetinaBody("Robot Link", {}, Pose((0,0,0),(0,0,0)),color=YELLOW,representation=AxisArrow())
    # bodies.append(b1)
    cube_dict = {
        0: (0,0,0),
        1: (0.1,0,0),
        2: (0,0.1,0),
        3: (0,0,0.1),
        4: (0.1,0.1,0),
        5: (0,0.1,0.1),
        6: (0.1,0,0.1),
        7: (0.1,0.1,0.1),
    }
    b2 = RetinaBody("Cube0", cube_dict, Pose((0,0,0), (0,0,0)), color=YELLOW)
    bodies.append(b2)

    # c0 = RetinaCamera(cs.WebcamStreamer(0, cs.mac_K, 0), None, get_cam_pose((0.5, 0.5, 0.5), (0, 0, 0)))
    # print(c0.pose.invert())
    # cameras.append(c0)

    bodies.append(RetinaBody("Carpet", representation=AxisRectangle("World", [GREEN, RED, BLUE, RED], top_left=(-0.5, 0.5, 0), width=1.0, height=1.0)))
    bodies.append(RetinaBody("Axes", representation=Axes()))


    world = World("World", bodies, cameras)

    writer = []
    while True:
        world.camera_pose = Pose(0,0,0.04,0,0,0) @ world.camera_pose
        world.bodies[0].pose = Pose(0,0,0.04,0,0,0.001) @ b2.pose
        new_frame = []
        world.display(new_frame)
        writer.append(new_frame)
        print(new_frame)