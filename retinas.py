import apriltag
import cv2
import retinas.utils.camera_streamer as cs
import numpy as np
from threading import Thread
import time
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt


from retinas.objects import RetinaCamera, RetinaBody
from retinas.pose import Pose, get_cam_pose
from retinas.test_bodies.cube_body import cube0_body, cube1_body
from retinas.test_bodies.world_body_4_corners import world_body
from retinas.utils.convex_hull import get_convex_hull_area
from retinas.world import World

DEFAULT_APRILTAG_DETECTOR = apriltag.Detector()
STRENGTH_CONSTANT = 1   # k

class ApriltagObserver:

    def __init__(self, camera_streamer, threshold=True):
        self.camera_streamer = camera_streamer
        self.threshold = threshold
        self.detector = apriltag.Detector(apriltag.DetectorOptions(
            families='tag36h11',
            border=1,
            nthreads=8,
            quad_decimate=1.0,
            quad_blur=0.0,
            refine_edges=True,
            refine_decode=False,
            refine_pose=False,
            debug=False,
            quad_contours=True
            ))

        self.grayscale_frame = None
        self.frame = None

    def get_observation(self):
        ret, frame = self.camera_streamer.read()

        if not ret:
            return [], np.empty((0,2))

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.threshold:
            # grayscale_frame = cv2.GaussianBlur(grayscale_frame, (3,3),0)
            _ , grayscale_frame = cv2.threshold(grayscale_frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        results = self.detector.detect(grayscale_frame)

        # print("After detect")
        tag_list = []
        label_list = []
        point_list = []
        for r in results:
            tag_list.append(r.tag_id)
            for corner in range(4):
                label_list.append((r.tag_id, corner))
                frame = cv2.circle(frame, np.array(r.corners[corner], dtype=np.int32), radius=0, color=(0,0, corner*63), thickness=16)
                point_list.append(r.corners[corner])
        # cv2.imshow(self.camera_streamer.name, grayscale_frame)
        # print("Showed")
        # print(tag_list)
        if len(tag_list) < 3:
            return self.get_observation()
        self.frame = frame
        self.grayscale_frame = grayscale_frame
        return label_list, np.array(point_list)


class Retinas(Thread):

    def __init__(self, observers, bodies, cameras, sticky=True):
        super().__init__()

        self.observers = observers
        self.bodies = bodies
        self.cameras = cameras
        self.J = len(observers)
        self.I = len(bodies)
        self.sticky = sticky

        self.tag_map = {}
        for body in bodies:
            point_dict = body.point_dict
            for label in point_dict:
                self.tag_map[label] = bodies.index(body)

        self.world_camera_poses = {}
        self.world_body_poses = {}
        self.__is__running__ = True
        self.start()

    def run(self):

        I = self.I
        J = self.J

        while self.__is__running__:
            # All the following are J x I
            N = {}
            A = {}
            T = {}
            E = {}
            G = {}
            if self.sticky:
                world_camera_poses = self.world_camera_poses
                world_body_poses = self.world_body_poses
            else:
                world_camera_poses = {}#self.world_camera_poses
                world_body_poses = {}#self.world_body_poses
            counter = 0
            for j in range(J):
                k_labels, k_points = self.observers[j].get_observation()
                
                # print(len(k_labels))
                for k in range(len(k_labels)):
                    label, point = k_labels[k], k_points[k]
                    i = self.tag_map[label]
                    if (j, i) in N:
                        N[j, i][0].append(label)
                        N[j, i][1].append(point)
                    else:
                        counter += 1
                        N[j, i] = [label], [point]

            for j, i in N:
                observer = self.observers[j]
                body = self.bodies[i]
                labels, points = N[j, i]

                A[j,i] = get_convex_hull_area(points)
                T[j,i] = self.do_pnp(labels, points, observer, body)
                E[j,i] = self.get_total_reprojection_error(labels, points, observer, body, T[j,i])[0]
                # print(E[j,i])
                temp = (len(N[j, i][0])**0.5) * E[j,i]
                G[j,i] = - np.log(1 + np.exp(-STRENGTH_CONSTANT) * temp)

            ####################
            nodes = ['b'+str(i) for i in range(I)] + ['c'+str(j) for j in range(J)]
            graph = nx.Graph()
            graph.add_nodes_from(nodes)

            for j, i in G:
                graph.add_edge('b'+str(i), 'c'+str(j))
                graph.add_edge('c'+str(j), 'b'+str(i))

            paths = nx.shortest_path(graph.to_undirected(), source='b0')
            
            for node in paths:
                path = paths[node]
                pose = Pose(0,0,0,0,0,0)
                cur = path[0]
                for step in path[1:]:
                    # print()
                    source = int(cur[1])
                    destin = int(step[1])
                    if cur[0] == 'c':
                        pose = pose @ T[source, destin]
                    elif cur[0] == 'b':
                        pose = pose @ T[destin, source].invert()
                    else:
                        raise Exception()
                    cur = step
                if node[0] == 'b':
                    world_body_poses[int(node[1])] = pose
                    self.bodies[int(node[1])].pose = pose
                if node[0] == 'c':
                    world_camera_poses[int(node[1])] = pose
                    self.cameras[int(node[1])].pose = pose
            ####################

            # for iteration in range(100):
            #     delta_world_body_poses = np.array((I, 6))
            #     for j in world_camera_poses:

            #         for i in world_body_poses:
            #             if (j, i) in T:
            #                 labels, points = N[j, i]
            #                 observer = self.observers[j]
            #                 body = self.bodies[i]
            #                 pose = world_body_poses[i].invert() @ world_camera_poses[j]
            #                 error, jacobian = cv2.projectPoints(np.array(visible_body[1]), np.array(points), K, D)
            #                 delta_T

            ####################

            for i, body in enumerate(self.bodies):
                if i not in world_body_poses:
                    world_body_poses[i] = None
                    self.bodies[i].pose = None
            for j, camera in enumerate(self.cameras):
                if j not in world_camera_poses:
                    world_camera_poses[j] = None
                    self.cameras[j].pose = None

            # print(world_body_poses)
            self.world_camera_poses = world_camera_poses
            self.world_body_poses = world_body_poses

    def get_total_reprojection_error(self, labels, points, observer, body, pose):
        visible_body = labels, []
        for label in labels:
            visible_body[1].append(body.point_dict[label])

        projected, jacobian = cv2.projectPoints(np.array(visible_body[1]), pose.rvec, pose.tvec, observer.camera_streamer.K, observer.camera_streamer.D)
        error = np.sum(np.power(np.squeeze(projected)-points, 2))
        jacobian_final = None
        return error, jacobian_final

    def do_pnp(self, labels, points, observer, body):
        visible_body = labels, []
        for label in labels:
            visible_body[1].append(body.point_dict[label])

        flag, rvec, tvec = cv2.solvePnP(np.array(visible_body[1]), np.array(points), observer.camera_streamer.K, observer.camera_streamer.D, flags=cv2.SOLVEPNP_EPNP)
        # print(Pose(rvec, tvec))
        # print(Pose(rvec, tvec).invert())
        return Pose(rvec, tvec)



if __name__ == '__main__':

    camera_streamers = []

    camera_streamers.append(cs.WebcamStreamer('rtsp://192.168.0.77:554', cs.iphone13_K))
    camera_streamers.append(cs.RemoteStreamer("http://192.168.0.222:8080/shot.jpg", cs.oneplus_8t_K))

    bodies = [world_body, cube0_body, cube1_body]

    cameras = [RetinaCamera(camera_streamer) for camera_streamer in camera_streamers]
    observers = [ApriltagObserver(camera_streamer) for camera_streamer in camera_streamers]

    world = World("My World", bodies, cameras)
    world.camera_pose = get_cam_pose((0.27, -1, 1),(0.27, 0.4, 0.3))
    retinas = Retinas(observers, bodies, cameras)

    while True:

        for j, streamer in enumerate(camera_streamers):
            if streamer.ret and (observers[j].frame is not None):
                cv2.imshow(f"camera frame {j}", observers[j].frame)

        world.display()