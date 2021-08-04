import apriltag
import cv2
import camera_tracking.camera_streamer as cs
import time
# import camera_tracking.pose as pose
import numpy as np

####################################################################

# DEFAULT CONFIGURATIONS:

import tagconfigs.link_config.config_12 as LC
import tagconfigs.world_config.config_12_45degrees as WC

LINK_TAG_DICT = LC.LINK_TAG_DICT
LINK_TAG_LENGTH = LC.LINK_TAG_LENGTH

CORNER_TAG_DICT = WC.CORNER_TAG_DICT
CORNER_TAG_LENGTH = WC.CORNER_TAG_LENGTH

GET_LINK_ID, GET_TAG_NUMBER = LC.get_link_id, LC.get_tag_number

####################################################################

# LABELING BEHAVIOR
SHOW_TAG_LABELS = True
SHOW_TAG_COORDS = True
SHOW_CORNER_TAG_LABELS = True


class TagDetector:

    def __init__(self, streamer, name, K, D):
        self.streamer = streamer
        self.name = name
        self.K = K
        self.D = D
        self.detector = apriltag.Detector()
        self.cTw = None
        self.frame = None
        cv2.namedWindow(self.name)

    @staticmethod
    def label_tag_coords(frame, r, tvec):
        (cX, cY) = (int(r.center[0]), int(r.center[1]))

        acc = 100

        cv2.putText(frame, "{}: [{}, {}, {}]".format(r.tag_id, int(tvec[0][0]*acc), int(tvec[1][0]*acc), int(tvec[2][0]*acc)), (cX, cY - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    @staticmethod
    def label_tag(frame, r):
        (cX, cY) = (int(r.center[0]), int(r.center[1]))

        cv2.putText(frame, "{}".format(r.tag_id), (cX, cY - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    @staticmethod
    def scale_image(frame, do_scale=True):
        if not do_scale:
            return frame
        scale = cs.IMAGE_WIDTH / frame.shape[1]
        new_width, new_height = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        return resized_frame

    def get_world_locs(self, wor_locs, label_tag=False,  label_coords=False, label_cor=False):
        ret, frame = self.streamer.read()
        cv2.circle(frame, (frame.shape[1]//2, frame.shape[0]//2), 5, (255, 255, 255), -1)

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(grayscale_frame)

        cam_lin_locs = {}
        cam_cor_locs = {}

        for r in results:
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            if r.tag_id in CORNER_TAG_DICT:
                tag_length = CORNER_TAG_LENGTH
                ret, rvec, tvec = pose.square_pnp_method(tag_length, r.corners, self.K, self.D)
                cam_cor_locs[r.tag_id] = tvec
            else:
                tag_length = LINK_TAG_LENGTH
                ret, rvec, tvec = pose.square_pnp_method(tag_length, r.corners, self.K, self.D)
                cam_lin_locs[r.tag_id] = tvec

        if label_tag:
            for r in results:
                td.label_tag(frame, r)

        if len(cam_cor_locs) < 3:
            self.frame = frame
            return

        cTw = pose.get_world_pose(cam_cor_locs, CORNER_TAG_DICT)

        self.cTw = cTw

        # print(cTw)

        if cTw is None:
            self.frame = frame
            return

        for tag_id in cam_lin_locs:
            wor_loc = (cTw @ np.append(cam_lin_locs[tag_id], np.array([[1]]), axis=0))[:3]
            if tag_id in wor_locs:
                wor_locs[tag_id].append(wor_loc)
            else:
                wor_locs[tag_id] = [wor_loc]

        if label_tag:
            for r in results:
                if label_coords:
                    if r.tag_id in cam_lin_locs:
                        loc = wor_locs[r.tag_id][-1]
                        # loc = cam_lin_locs[r.tag_id]
                        td.label_tag_coords(frame, r, loc)
                    elif label_cor:
                        loc = (cTw @ np.append(cam_cor_locs[r.tag_id], np.array([[1]]), axis=0))[:3]
                        # loc = cam_cor_locs[r.tag_id]
                        td.label_tag_coords(frame, r, loc)

        self.frame = frame


if __name__ == '__main__':

    tds = []

    streamer1 = cs.RemoteStreamer(cs.URL, cs.oneplus_8t_K)
    time.sleep(2)
    td1 = TagDetector(streamer1, "Phone", streamer1.K, 0)
    tds.append(td1)

    streamer2 = cs.WebcamStreamer(0, cs.mac_K)
    td2 = TagDetector(streamer2, "ELP", streamer2.K, 0)
    tds.append(td2)

    # streamer3 = cs.WebcamStreamer(0, cs.mac_K)
    # td3 = TagDetector(streamer3, "Mac", streamer3.K, 0)
    # tds.append(td3)

    while True:
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        wor_locs = {}

        for td in tds:
            td.get_world_locs(wor_locs, SHOW_TAG_LABELS, SHOW_TAG_COORDS, SHOW_CORNER_TAG_LABELS)

        link_poses = pose.get_link_poses(wor_locs, LINK_TAG_DICT, GET_LINK_ID, GET_TAG_NUMBER)
        for td in tds:
            if td.cTw is None:
                print(td.name)
                continue
            for link in link_poses:
                pose.draw_link_pose(td.frame, link_poses[link], td.cTw, td.K)
            cv2.imshow(td.name, td.scale_image(td.frame))

    for i in range(1, 5):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    for td in tds:
        td.streamer.close()
    exit()
