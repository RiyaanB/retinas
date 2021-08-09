import cv2
import numpy as np

PRINT_MODE = "RADIANS"


class Pose:

    @staticmethod
    def get_4x4_from_vectors(rvec, tvec):
        R = cv2.Rodrigues(np.array(rvec, dtype=float).reshape(3, 1))[0]
        matrix = np.zeros((4, 4))
        matrix[:3, :3] = R
        matrix[0][3], matrix[1][3], matrix[2][3] = tvec
        matrix[3, :] = [0, 0, 0, 1.]
        return matrix

    @staticmethod
    def get_vectors_from_4x4(matrix):
        rvec = cv2.Rodrigues(matrix[:3, :3])[0].ravel()
        rvec = rvec[0], rvec[1], rvec[2]
        tvec = matrix[0][3], matrix[1][3], matrix[2][3]
        return rvec, tvec

    def __init__(self, rvec, tvec=None, r2=None, t1=None, t2=None, t3=None):

        if isinstance(rvec, Pose):
            self.__init__(rvec.rvec, rvec.tvec)
        elif t3 is not None:
            self.__init__((rvec, tvec, r2), (t1, t2, t3))
        elif tvec is None:
            if isinstance(rvec, np.ndarray):
                self.matrix = rvec
                self.rvec, self.tvec = Pose.get_vectors_from_4x4(self.matrix)
            else:
                assert len(rvec) == 2
                self.__init__(rvec[0], rvec[1])
        else:
            if isinstance(rvec, np.ndarray):
                rvec = rvec.ravel()
            if isinstance(tvec, np.ndarray):
                tvec = tvec.ravel()
            self.rvec = rvec[0], rvec[1], rvec[2]
            self.tvec = tvec[0], tvec[1], tvec[2]
            self.matrix = Pose.get_4x4_from_vectors(self.rvec, self.tvec)

    def __matmul__(self, other):
        return Pose(self.matrix @ other.matrix)

    def invert(self):
        return Pose(np.linalg.inv(self.matrix))

    def __str__(self):
        if PRINT_MODE == "RADIANS":
            return "({}, {}, {}), ({}, {}, {})".format(*self.rvec, *self.tvec)
        else:
            norm = np.linalg.norm(self.rvec)
            factor = (1./norm)
            angle = norm*(180/np.pi)
            return "{}deg ({}, {}, {}), ({}, {}, {})".format(angle, *(np.array(self.rvec)*factor), *self.tvec)


def get_cam_pose(position, target):
    # returns world pose of camera
    position = np.array([position], dtype=float)
    target = np.array([target], dtype=float)
    # GRAM-SCHMIDT PROCESS
    direction_vector = target - position
    unit_direction_vector = direction_vector / np.linalg.norm(direction_vector)
    up_vector = np.array([[0, 0, 1]])

    z_hat = unit_direction_vector
    x = (np.cross(z_hat, up_vector))
    if np.linalg.norm(x) == 0:
        x = get_cam_pose.PREV_X
    else:
        get_cam_pose.PREV_X = x

    x_hat = x / np.linalg.norm(x)
    y_hat = np.cross(z_hat, x_hat)

    cam_pose = np.zeros((4, 4))

    # print(x_hat.T)
    # print(y_hat.T)
    # print(z_hat.T)
    # print(position.T)

    cam_pose[:3, 0:1] = x_hat.T
    cam_pose[:3, 1:2] = y_hat.T
    cam_pose[:3, 2:3] = z_hat.T
    cam_pose[:3, 3:4] = position.T
    cam_pose[3, :] = [0, 0, 0, 1]

    return Pose(cam_pose)


get_cam_pose.PREV_X = np.array([[1, 0, 0]])

if __name__ == '__main__':
    PRINT_MODE = "DEGREES"
    a = get_cam_pose((0, -1., 0), (-1, 0, 0)).matrix
    print(a)
