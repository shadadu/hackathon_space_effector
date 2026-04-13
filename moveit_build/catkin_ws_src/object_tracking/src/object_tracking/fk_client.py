import rospy
import numpy as np
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest

import rospy
import moveit_msgs.srv
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotState

import rospy
from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest
from moveit_msgs.msg import PlanningSceneComponents


def get_panda_joint_positions():
    # 1. Wait for the service to be available
    rospy.wait_for_service('/get_planning_scene')

    try:
        # 2. Create the ServiceProxy
        get_scene = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)

        # 3. Request only the robot state component
        request = GetPlanningSceneRequest()
        request.components.components = PlanningSceneComponents.ROBOT_STATE

        # 4. Call the service
        response = get_scene(request)

        # 5. Extract Joint Positions
        joint_state = response.scene.robot_state.joint_state
        # Panda joints are typically named panda_joint1 ... panda_joint7
        print("Joint Positions:", joint_state.position)
        return joint_state.position

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
#
#
# if __name__ == "__main__":
#     rospy.init_node('get_panda_joints_client')
#     get_panda_joint_positions()


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

    def close(self):
        self.fk.close()

