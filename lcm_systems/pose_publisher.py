import lcm
from lcm_systems.lcm_types.lcm_pose import lcmt_object_state
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class PosePublisher:
    def __init__(self):
        self.lc = lcm.LCM()
        
    def publish_pose(self, obj_name = "OBJECT", pose = None):
        # Instantiate the message object
        self.pose_msg = lcmt_object_state()
        # Set the message fields
        self.pose_msg.utime = int(time.time() * 1000000)
        self.pose_msg.object_name = obj_name
        self.pose_msg.num_positions = 7
        self.pose_msg.num_velocities = 6
        self.pose_msg.position_names = ["capsule_1_qw", "capsule_1_qx", "capsule_1_qy", "capsule_1_qz", "capsule_1_x", "capsule_1_y", "capsule_1_z"]
        self.pose_msg.velocity_names = ["capsule_1_wx", "capsule_1_wy", "capsule_1_wz", "capsule_1_vx", "capsule_1_vy", "capsule_1_vz"]
        # Convert the pose to a 7-element array
        self.pose_msg.position = self.homogeneous_matrix_to_pose(pose)
        self.pose_msg.velocity = np.zeros(6)
        self.lc.publish("OBJECT_STATE", self.pose_msg.encode())

    def homogeneous_matrix_to_pose(self, homogeneous_matrix):
        # Convert a 4x4 homogeneous matrix to a 7 element pose with quaternion
        pose = np.zeros(7)
        scipy_quat = R.from_matrix(homogeneous_matrix[:3,:3]).as_quat()
        pose[0] = scipy_quat[-1]
        pose[1:4] = scipy_quat[:3]
        pose[4:] = homogeneous_matrix[:3,3]
        return pose
