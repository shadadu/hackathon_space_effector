import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from moveit_commander import PlanningSceneInterface
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

class ObjectPlanningSceneUpdater:

    def __init__(self):
        rospy.init_node("object_to_planning_scene")

        self.scene = PlanningSceneInterface(synchronous=True)
        rospy.sleep(2)

        self.object_id = "free_object"
        self.prediction_dt = 0.3  # seconds ahead to compensate latency

        rospy.Subscriber("/object/state", Odometry, self.state_callback)

        rospy.loginfo("Object Planning Scene Updater Ready.")
        rospy.spin()

    def state_callback(self, msg):

        # Extract pose and velocity
        pose = msg.pose.pose
        twist = msg.twist.twist

        # -----------------------------
        # Microgravity Prediction
        # x(t+dt) = x + v*dt
        # -----------------------------

        predicted_pose = PoseStamped()
        predicted_pose.header.frame_id = "world"
        predicted_pose.header.stamp = rospy.Time.now()

        predicted_pose.pose.position.x = pose.position.x + twist.linear.x * self.prediction_dt
        predicted_pose.pose.position.y = pose.position.y + twist.linear.y * self.prediction_dt
        predicted_pose.pose.position.z = pose.position.z + twist.linear.z * self.prediction_dt

        predicted_pose.pose.orientation = pose.orientation

        self.update_collision_object(predicted_pose)

    def update_collision_object(self, pose_stamped):

        co = CollisionObject()
        co.id = self.object_id
        co.header.frame_id = "world"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.2, 0.2, 0.2]

        co.primitives = [primitive]
        co.primitive_poses = [pose_stamped.pose]
        co.operation = CollisionObject.ADD

        self.scene._pub_co.publish(co)


if __name__ == "__main__":
    ObjectPlanningSceneUpdater()
