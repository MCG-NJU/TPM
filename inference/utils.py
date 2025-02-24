import datetime
import geometry_msgs.msg
import os
from moveit_commander.conversions import pose_to_list
from math import fabs, cos, sqrt


KINECT_SERIAL_NUMBER_MAPPING = {
    '0': '000424924212',
    '1': '000334924212'
}


def check_grasp_data():
    import os
    import numpy as np

    p = '/Data/grasp_data/'

    tar = []
    for i in os.listdir(p):
        if i[0] == '-':
            tar.append(i)

    print(f'data count: {len(tar)}')

    suffix_0 = '/images/view_0/'
    suffix_1 = '/images/view_1/'

    for session_name in tar:
        p0 = p + session_name + suffix_0
        p1 = p + session_name + suffix_1

        dp = p + session_name + '/data.npy'
        data = np.load(open(dp, 'rb'))
        print(session_name)
        print(len(data), len(os.listdir(p0)), len(os.listdir(p1)))


def get_cur_dir():
    # Mind that this function only returns the directory of THIS FILE,
    # not the file that calls this function
    return os.path.dirname(os.path.abspath(__file__))

def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')

def get_datestamp():
    return datetime.datetime.now().strftime('%m%d')

def dist(p, q):
    return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True