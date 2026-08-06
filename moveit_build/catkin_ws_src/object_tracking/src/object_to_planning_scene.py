#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry

from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


class ObjectPlanningSceneUpdater:
    def __init__(self):
        self.object_topic = rospy.get_param("~object_topic", "/object/state")
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.object_id = rospy.get_param("~object_id", "free_object")

        # Simple object geometry (box) — tune later
        self.size_x = rospy.get_param("~size_x", 0.06)
        self.size_y = rospy.get_param("~size_y", 0.06)
        self.size_z = rospy.get_param("~size_z", 0.12)

        # If we haven't received state yet, keep object far away to avoid collisions
        self.spawn_far_x = rospy.get_param("~spawn_far_x", 1.0)

        # Public, stable topic used by MoveIt to receive collision objects
        self.pub_co = rospy.Publisher("/collision_object", CollisionObject, queue_size=10)

        self.last_pose = None
        rospy.Subscriber(self.object_topic, Odometry, self.state_callback, queue_size=1)

        rospy.loginfo("Object Planning Scene Updater Ready. Publishing to /collision_object (id=%s)", self.object_id)

        # Send an initial far-away object so planning starts collision-free
        self.publish_object(self.make_far_pose())

    def make_far_pose(self):
        p = Pose()
        p.position.x = self.spawn_far_x
        p.position.y = 0.0
        p.position.z = 0.0
        p.orientation.w = 1.0
        return p

    def state_callback(self, msg: Odometry):
        # Use incoming pose directly (already in msg.header.frame_id, typically 'world')
        pose = msg.pose.pose

        # If frame differs, you can enforce a frame id param; for now trust msg.header.frame_id
        frame = msg.header.frame_id if msg.header.frame_id else self.frame_id
        self.publish_object(pose, frame_id=frame)

    def publish_object(self, pose: Pose, frame_id: str = None):
        if frame_id is None:
            frame_id = self.frame_id

        co = CollisionObject()
        co.header.stamp = rospy.Time.now()
        co.header.frame_id = frame_id
        co.id = self.object_id
        co.operation = CollisionObject.ADD

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [self.size_x, self.size_y, self.size_z]

        co.primitives = [primitive]
        co.primitive_poses = [pose]

        self.pub_co.publish(co)


def main():
    rospy.init_node("object_to_planning_scene")
    ObjectPlanningSceneUpdater()
    rospy.spin()


if __name__ == "__main__":
    main()
