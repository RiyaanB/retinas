import apriltag
import cv2
import utils.camera_streamer as cs
import numpy as np
from threading import Thread
import time

from objects import RetinaCamera
from pose import Pose, get_cam_pose
from test_bodies.cube_body import cube0_body, cube1_body
from test_bodies.world_body_4_corners import world_body
from utils.convex_hull import get_convex_hull_area
from world import World

DEFAULT_APRILTAG_DETECTOR = apriltag.Detector()
STRENGTH_CONSTANT = 1   # k


class Observer:

    def __init__(self, camera_streamer):
        self.camera_streamer = camera_streamer

    def get_observation(self):
        pass


class ApriltagObserver(Observer):

    def __init__(self, camera_streamer, detector=DEFAULT_APRILTAG_DETECTOR):
        super().__init__(camera_streamer)
        self.detector = detector

    def get_observation(self):
        ret, frame = self.camera_streamer.read()
        if not ret:
            return [], np.array([])
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(grayscale_frame)
        label_list = []
        point_list = []
        for r in results:
            tag_id = r.tag_id
            for corner in range(4):
                label_list.append((tag_id, corner))
                point_list.append(r.corners[corner])
        # print("friwyf", label_list)
        return label_list, np.array(point_list)


class Retinas(Thread):

    def __init__(self, observers, bodies, tag_map=None):
        super().__init__()

        self.observers = observers
        self.bodies = bodies
        self.J = len(observers)
        self.I = len(bodies)

        if tag_map is None:
            tag_map = {}
            for body in bodies:
                point_dict = body.point_dict
                for label in point_dict:
                    tag_map[label] = bodies.index(body)
        self.tag_map = tag_map

        self.world_camera_poses = {}
        self.world_body_poses = {}
        self.__is__running__ = True
        self.start()

    def run(self):
        I = self.I
        J = self.J

        while self.__is__running__:
            N = {}  # IxJ --> [LxP]
            A = {}  # IxJ --> R
            T = {}  # IxJ --> R^6
            E = {}  # IxJ --> R
            G = {}  # IxJ --> R
            graph = {}  # IxJ --> [0,1]
            # print("hunumunu")
            for j in range(J):
                cj_observations = self.observers[j].get_observation()
                # if not len(cj_observations[0]) == len(cj_observations[1]):
                # print(f'Labels: {(cj_observations[0])}')
                # print(f'Points: {(cj_observations[1].shape)}')
                for observation_number in range(len(cj_observations[0])):
                    label = cj_observations[0][observation_number]
                    point = cj_observations[1][observation_number]
                    # print(point)
                    i = self.tag_map[label]
                    if (i, j) in N:
                        N[i, j][0].append(label)
                        N[i, j][1].append(point)
                    else:
                        N[i, j] = [label], [point]


                for i, j in N:
                    observer = self.observers[j]
                    body = self.bodies[i]
                    labels, points = N[i, j]
                    A[i, j] = get_convex_hull_area(points)
                    T[i, j] = self.do_pnp(labels, points, observer, body)
                    E[i, j] = self.get_total_reprojection_error(labels, points, observer, body, T[i, j])
                    G[i, j] = (len(N[i, j]) ** 0.5) * A[i, j] * E[i, j]
                    graph[i, j] = - np.log(1 + np.exp(-STRENGTH_CONSTANT * G[i, j]))

            mst = self.get_mst(graph)
            pose_observations = []
            for i in range(I):
                row = []
                for j in range(J):
                    row.append(None)
                pose_observations.append(row)

            self.T = T
            for i, j in mst:
                pose_observations[i][j] = T[i, j]

            world_camera_poses = {}
            world_body_poses = {}

            fringe_i = [(0, None)]
            fringe_j = []
            while not len(fringe_i) + len(fringe_j) == 0:
                if len(fringe_i) > 0:
                    i, parent = fringe_i.pop(-1)

                    if parent is None:
                        world_body_poses[i] = Pose(0,0,0,0,0,0)
                    else:
                        world_body_poses[i] = world_camera_poses[parent] @ pose_observations[i][parent] # PROBABLY WRONG

                    for j in range(J):
                        if (pose_observations[i][j] is not None) and (j not in world_camera_poses):
                            fringe_j.append((j, i))
                if len(fringe_j) > 0:
                    j, parent = fringe_j.pop(-1)

                    # print(f'{j}  {world_body_poses[parent]}  {pose_observations[parent][j]}')

                    world_camera_poses[j] = pose_observations[parent][j] @ world_body_poses[parent]    # PROBABLY WRONG

                    for i in range(I):
                        if (pose_observations[i][j] is not None) and (i not in world_body_poses):
                            fringe_i.append((i, j))

            self.world_camera_poses = world_camera_poses
            self.world_body_poses = world_body_poses
            for i in range(len(bodies)):
                if i in world_body_poses:
                    bodies[i].pose = world_body_poses[i]
                else:
                    bodies[i].pose = Pose(0,0,0,10000,10000,10000)
            for j in range(len(cameras)):
                if j in world_camera_poses:
                    cameras[j].pose = world_camera_poses[j]
                else:
                    cameras[j].pose = Pose(0,0,0,10000,10000,10000)

    def get_mst(self, graph):
        edges = list(graph.items())
        edges.sort(key=lambda x: x[1], reverse=True)
        visited = {}
        mst = {}
        for source_dest, weight in edges:
            if source_dest[1] not in visited:
                mst[source_dest] = weight
        return mst

    def get_total_reprojection_error(self, labels, points, observer, body, pose):
        visible_body = labels, []
        for label in labels:
            visible_body[1].append(body.point_dict[label])
            # for l in range(len(body[1])):
            #     if body[0][l] == label:
            #         visible_body[1].append(body[1][l])
        projected, _ = cv2.projectPoints(np.array(visible_body[1]), pose.rvec, pose.tvec, observer.camera_streamer.K, observer.camera_streamer.D)
        return np.power(np.sum(np.power(projected - points, 2)), 0.5)

    def do_pnp(self, labels, points, observer, body):
        print(labels)
        print(points)
        visible_body = labels, []
        for label in labels:
            visible_body[1].append(body.point_dict[label])
            # for l in range(len(body[1])):
            #     if body[0][l] == label:
            #         visible_body[1].append(body[1][l])
        # print(np.array(points))
        flag, rvec, tvec = cv2.solvePnP(np.array(visible_body[1]), np.array(points), observer.camera_streamer.K, observer.camera_streamer.D, flags=cv2.SOLVEPNP_EPNP)
        # print(f'{observer.camera_streamer.name} {body.name} {Pose(rvec, tvec)}')
        return Pose(rvec, tvec)


if __name__ == '__main__':
    # DEBUG = False
    # streamer = cs.WebcamStreamer(0, cs.mac_K)
    streamer = cs.RemoteStreamer(cs.URL, cs.oneplus_8t_K)
    observer = ApriltagObserver(streamer)
    observers = [observer]
    bodies = [world_body, cube0_body, cube1_body]

    cameras = []
    cameras.append(RetinaCamera(observer.camera_streamer, None, get_cam_pose((0.25, -0.5, 0.5), (0.25, 0.25, 0))))

    my_retinas = Retinas(observers, bodies)
    # while True:
    #     if 0 in my_retinas.world_camera_poses:
    #         # print(my_retinas.world_camera_poses[0])
    #         time.sleep(0.1)

    world = World("World", bodies, cameras, camera_pose=get_cam_pose((0.25, -0.5, 2), (0.25, 0.25, 0)))

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

    my_retinas.__is__running__ = False

    for a in range(30):
        cv2.destroyWindow(world.name)

    exit()
    quit()