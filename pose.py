import cv2
import numpy as np


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
        return "({}, {}, {}), ({}, {}, {})".format(*self.rvec, *self.tvec)

