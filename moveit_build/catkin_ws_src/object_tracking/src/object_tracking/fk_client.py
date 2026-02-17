import rospy
import numpy as np
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest


class FKClient:
    def __init__(self, service="/compute_fk", ee_link="panda_hand", frame="world", timeout=30.0):
        self.service = service
        self.ee_link = ee_link
        self.frame = frame
        rospy.wait_for_service(service, timeout=timeout)
        self.fk = rospy.ServiceProxy(service, GetPositionFK)

    def ee_position(self, joint_names, q):
        req = GetPositionFKRequest()
        req.header.frame_id = self.frame
        req.fk_link_names = [self.ee_link]
        req.robot_state.joint_state.name = list(joint_names)
        req.robot_state.joint_state.position = [float(x) for x in q]
        resp = self.fk(req)
        if resp.error_code.val != 1 or not resp.pose_stamped:
            raise RuntimeError(f"FK failed: code={resp.error_code.val}")
        p = resp.pose_stamped[0].pose.position
        return np.array([p.x, p.y, p.z], dtype=np.float64)
