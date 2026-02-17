import rospy
import numpy as np
from jacobian_server.srv import GetJacobian, GetJacobianRequest


class JacobianClient:
    def __init__(self, service="/get_jacobian", timeout=30.0):
        rospy.wait_for_service(service, timeout=timeout)
        self.jac = rospy.ServiceProxy(service, GetJacobian)

    def jacobian(self, group_name, link_name, joint_names, q):
        req = GetJacobianRequest()
        req.group_name = group_name
        req.link_name = link_name
        req.joint_names = list(joint_names)
        req.joint_positions = [float(x) for x in q]
        req.reference_point.x = 0.0
        req.reference_point.y = 0.0
        req.reference_point.z = 0.0
        resp = self.jac(req)
        # resp.jacobian is typically a flat array or matrix-like message depending on your srv
        return resp