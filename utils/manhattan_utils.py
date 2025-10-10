import math
import numpy as np
def create_man_rans(position, rotation):
    # create manhattan transformation matrix for threejs
    # The angle is reversed because the counterclockwise direction is defined as negative in three.js
    rot_x = np.array([[1, 0, 0],
                      [0, math.cos(np.deg2rad(-rotation[0])), -math.sin(np.deg2rad(-rotation[0]))],
                      [0, math.sin(np.deg2rad(-rotation[0])),  math.cos(np.deg2rad(-rotation[0]))]])
    rot_y = np.array([[ math.cos(np.deg2rad(-rotation[1])), 0, math.sin(np.deg2rad(-rotation[1]))],
                      [0, 1, 0],
                      [-math.sin(np.deg2rad(-rotation[1])), 0, math.cos(np.deg2rad(-rotation[1]))]])
    rot_z = np.array([[math.cos(np.deg2rad(-rotation[2])), -math.sin(np.deg2rad(-rotation[2])), 0],
                      [math.sin(np.deg2rad(-rotation[2])),  math.cos(np.deg2rad(-rotation[2])), 0],
                      [0, 0, 1]])

    rot = rot_z @ rot_y @ rot_x
    man_trans = np.zeros((4, 4))
    man_trans[:3, :3] = rot.transpose()
    man_trans[:3, -1] = np.array(position).transpose()
    man_trans[3, 3] = 1

    return man_trans


def get_man_trans(pos,rot):
    pos = [float(pos) for pos in pos.split(" ")]
    rot = [float(rot) for rot in rot.split(" ")]
    man_trans = create_man_rans(pos, rot)
    return man_trans