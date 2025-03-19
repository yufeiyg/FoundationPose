import lcm
from lcm_systems.lcm_types.lcm_pose import lcmt_object_state
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class PosePublisher:
    def __init__(self, system_name: str):
        self.lc = lcm.LCM()
        assert system_name in ['jack', 't']
        self.prefix = 'capsule_1' if system_name=='jack' else 'vertical_link'
        
    def publish_pose(self, obj_name = "OBJECT", pose = None):
        # Instantiate the message object
        self.pose_msg = lcmt_object_state()
        # Set the message fields
        self.pose_msg.utime = int(time.time() * 1000000)
        self.pose_msg.object_name = obj_name
        self.pose_msg.num_positions = 7
        self.pose_msg.num_velocities = 6
        self.pose_msg.position_names = [
            f"{self.prefix}_qw", f"{self.prefix}_qx", f"{self.prefix}_qy",
            f"{self.prefix}_qz", f"{self.prefix}_x", f"{self.prefix}_y",
            f"{self.prefix}_z"]
        self.pose_msg.velocity_names = [
            f"{self.prefix}_wx", f"{self.prefix}_wy", f"{self.prefix}_wz",
            f"{self.prefix}_vx", f"{self.prefix}_vy", f"{self.prefix}_vz"]
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
