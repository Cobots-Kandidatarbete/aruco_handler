import rclpy
from rclpy.node import Node
from tf_tools_msgs.srv import LookupTransform
from tf_tools_msgs.srv import ManipulateScene
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger
from rclpy.executors import MultiThreadedExecutor
import threading


class Triggers:
    trigger_locking = False
    call_done = False


class ServiceNode(Node):
    def __init__(self):
        super().__init__("aruco_locker_service_node")

        self.srv = self.create_service(
            Trigger, 'lock_arucos', self.srv_callback)

        self.get_logger().info(f"aruco_locker service node should be running")

    def srv_callback(self, request, response):
        ac = ArucoLocker()
        ac.lock_markers()
        response.success = True
        return response

# The idea is to ask the lookup for the position of a marker in the logitech_270
# frame, and then ask the scene manipulation service to update the state in the
# broadcaster by adding a static 'locked' frame of the marker so that the robot
# can move again without the marker moving with it. So save the locked in world.


class ArucoLocker(Node):
    def __init__(self):
        super().__init__("aruco_locker")

        self.transform = TransformStamped()

        self.lookup_client = self.create_client(LookupTransform, "tf_lookup")
        self.lookup_request = LookupTransform.Request()
        self.lookup_response = LookupTransform.Response()

        self.sms_client = self.create_client(
            ManipulateScene, "manipulate_scene")
        self.sms_request = ManipulateScene.Request()
        self.sms_response = ManipulateScene.Response()

        while not self.lookup_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("tf lookup service not available, waiting again...")

        while not self.sms_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("sms service not available, waiting again...")

        self.get_logger().info(f"aruco_locker should be running")

    def lock_markers(self):
        for id in range(0, 2):
            self.lock_a_marker(id)

    def lock_a_marker(self, id):
        self.lookup_request.parent_id = "world"
        self.lookup_request.child_id = f"aruco_{id}"
        self.lookup_request.deadline = 2000
        self.lookup_future = self.lookup_client.call_async(self.lookup_request)
        self.get_logger().info(f"request sent: {self.lookup_request}")
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.lookup_future.done():
                try:
                    self.lookup_response = self.lookup_future.result()
                except Exception as e:
                    self.get_logger().error(
                        f"service call failed with: {(e,)}")
                else:
                    self.get_logger().info(
                        f"lookup result: {self.lookup_response}")
                finally:
                    self.get_logger().info(f"service call completed")
                break

        if self.lookup_response != None:
            if self.lookup_response.success:
                self.sms_request.command = "update"
                self.sms_request.parent_frame = "world"
                self.sms_request.child_frame = f"locked_aruco_{id}"
                self.sms_request.transform = self.lookup_response.transform.transform
                self.sms_request.same_position_in_world = False
                self.sms_future = self.sms_client.call_async(self.sms_request)
                self.get_logger().info(f"request sent: {self.sms_request}")
                while rclpy.ok():
                    rclpy.spin_once(self)
                    if self.sms_future.done():
                        try:
                            self.sms_response = self.sms_future.result()
                        except Exception as e:
                            self.get_logger().error(
                                f"service call failed with: {(e,)}")
                        else:
                            self.get_logger().info(
                                f"lookup result: {self.sms_response}")
                        finally:
                            self.get_logger().info(f"service call completed")
                        break


def main(args=None):
    rclpy.init(args=args)
    try:
        c1 = ArucoLocker()
        c2 = ServiceNode()

        executor = MultiThreadedExecutor()
        executor.add_node(c1)
        executor.add_node(c2)

        try:
            executor.spin()
        finally:
            executor.shutdown()
            c1.destroy_node()
            c2.destroy_node()

    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
