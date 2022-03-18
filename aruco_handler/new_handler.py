import numpy as np
import threading
import cv2
import transforms3d

import os

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Transform, TransformStamped
from builtin_interfaces.msg import Time
from tf2_msgs.msg import TFMessage

class StampContainer:
    aruco_stamps = []

    @classmethod
    def clear(cls):
        cls.aruco_stamps = []


class DrawConfig:
    box_color = 0, 255, 0
    circle_color = 0, 0, 255
    box_thickness = 2
    circle_radius = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = 0, 255, 0
    font_scale = 0.5
    font_thickness = 2
    axis_length = 0.15


class CameraConfig:
    aruco_dict_key = cv2.aruco.DICT_6X6_50

    capture = None
    aruco_dict = None
    aruco_params = None

    marker_size = 0.05
    camera_name = "realsense_d435"
    camera_index = 6
    marker_size = 0.05

    camera_matrix = None
    distortion_coeffecients = None
    optimal_camera_matrix = None
    image_size = None
    rectification_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float')
    map_1_type = cv2.CV_32F

    map_x, map_y = None, None

    @classmethod
    def load_calibration_file(cls):
        path_to_calibrator_dir = f"{os.path.curdir}/src/camera-calibrator/{cls.camera_name}_configs"
        if not os.path.isdir(path_to_calibrator_dir):
            second_path = f"{os.path.pardir}/camera-calibrator/{cls.camera_name}_configs"
            if os.path.isdir(second_path):
                path_to_calibrator_dir = second_path
            else:
                raise ImportError(f"Could not find dir {path_to_calibrator_dir} or {second_path}")

        path_to_calibration_file = f"{path_to_calibrator_dir}/{cls.camera_name}_config.yaml"
        if not os.path.isfile(path_to_calibration_file):
            raise ImportError(f"Could not find file {path_to_calibration_file}")

        try:
            calibration_file = cv2.FileStorage(path_to_calibration_file, cv2.FILE_STORAGE_READ)
            cls.camera_matrix = calibration_file.getNode('camera_matrix').mat()
            cls.distortion_coeffecients = calibration_file.getNode('distortion_matrix').mat()
            cls.optimal_camera_matrix = calibration_file.getNode('optimal_camera_matrix').mat()
            cls.image_size = calibration_file.getNode('image_size').mat().astype(int).flatten()

        finally:
            calibration_file.release()

    @classmethod
    def calibrate_camera(cls):
        cls.load_calibration_file()
        
        cls.capture = cv2.VideoCapture(cls.camera_index)
        cls.aruco_dict = cv2.aruco.Dictionary_get(cls.aruco_dict_key)
        cls.aruco_params = cv2.aruco.DetectorParameters_create()

        cls.map_x, cls.map_y = cv2.initUndistortRectifyMap(
            cls.camera_matrix,
            cls.distortion_coeffecients,
            cls.rectification_matrix,
            cls.camera_matrix,
            cls.image_size,
            cls.map_1_type
        )
        
        cv2.ShowUndistortedImage = True


class ArUcoTracker(Node):
    def __init__(self):
        super().__init__(node_name="aruco_tracker")

        CameraConfig.calibrate_camera()

        self.get_logger().info("aruco_tracker node should be started")
        t1 = threading.Thread(target=self.run_vision_callback)
        t1.daemon = True
        t1.start()

    def run_vision_callback(self):
        while True:
            
            capture_success, frame = CameraConfig.capture.read()
            if not capture_success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.remap(frame, CameraConfig.map_x, CameraConfig.map_y, cv2.INTER_LINEAR)
            corners, ids, _ = cv2.aruco.detectMarkers(frame, CameraConfig.aruco_dict, parameters=CameraConfig.aruco_params)

            if corners:
                corners = np.array(corners)
                ids = ids.flatten()

                for marker_corner, marker_id in zip(corners, ids):
                    rotation, translation, transform = self.get_pose_vectors(marker_corner)
                    self.buffer_marker(marker_id, transform)

                    self.draw_marker_info(frame, marker_corner, marker_id, rotation, translation)


            cv2.imshow('img', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        CameraConfig.capture.release()
        cv2.destroyAllWindows()

    def buffer_marker(self, marker_id, transform):
        transform_stamped = self.create_transform_stamp(marker_id, transform)
        StampContainer.aruco_stamps.append(transform_stamped)

    def get_pose_vectors(self, marker_corner):
        # http://amroamroamro.github.io/mexopencv/matlab/cv.estimatePoseSingleMarkers.html
        pose_estimation = cv2.aruco.estimatePoseSingleMarkers(marker_corner, 
        CameraConfig.marker_size, CameraConfig.camera_matrix, CameraConfig.distortion_coeffecients)
                    
        rotation = pose_estimation[0][0, 0, :]
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        rotation_quaternion = transforms3d.quaternions.mat2quat(rotation_matrix)

        translation = pose_estimation[1][0, 0, :]

        transform = self.get_pose_transform(rotation_quaternion, translation)
        return rotation,translation,transform

    def create_transform_stamp(self, marker_id, transform):
        transform_stamped = TransformStamped()
        transform_stamped.header.frame_id = CameraConfig.camera_name
        transform_stamped.header.stamp = Time()
                    
        current_time = self.get_clock().now().seconds_nanoseconds()
                    
        transform_stamped.header.stamp.sec = current_time[0]
        transform_stamped.header.stamp.nanosec = current_time[1]

        transform_stamped.child_frame_id = f"aruco_{marker_id}"
        transform_stamped.transform = transform
        return transform_stamped

    def get_pose_transform(self, rotation_quaternion, translation):
        transform = Transform()
        transform.translation.x, transform.translation.y, transform.translation.z = translation
        transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z = rotation_quaternion
        print(transform)
        return transform

    def draw_marker_info(self, frame, marker_corner, marker_id, rotation, translation):
        top_left, top_right, bot_right, bot_left = marker_corner.reshape((4, 2)).astype(int)

        cv2.line(frame, top_left, top_right, DrawConfig.box_color, DrawConfig.box_thickness)
        cv2.line(frame, top_right, bot_right, DrawConfig.box_color, DrawConfig.box_thickness)
        cv2.line(frame, bot_right, bot_left, DrawConfig.box_color, DrawConfig.box_thickness)
        cv2.line(frame, bot_left, top_left, DrawConfig.box_color, DrawConfig.box_thickness)

        center = (top_left[0] + bot_right[0]) // 2, (top_left[1] + bot_right[1]) // 2
        cv2.circle(frame, center, DrawConfig.circle_radius, DrawConfig.circle_color, -1)

        cv2.putText(frame, str(marker_id), (top_left[0], top_right[1] - 15), DrawConfig.font, DrawConfig.font_scale, DrawConfig.font_color, DrawConfig.font_thickness)
        cv2.aruco.drawAxis(frame, CameraConfig.camera_matrix, CameraConfig.distortion_coeffecients, rotation, translation, DrawConfig.axis_length)


class ArUcoPublisher(Node):
    def __init__(self):
        super().__init__("aruco_publisher")

        history_depth = 20
        self.tf_publisher = self.create_publisher(TFMessage, "/tf", history_depth)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("aruco_tracker should be started.")

    def timer_callback(self):
        messages = [a for a in StampContainer.aruco_stamps]
        
        tf_message = TFMessage()
        tf_message.transforms = messages

        self.tf_publisher.publish(tf_message)

        StampContainer.clear()


def main(args=None):
    rclpy.init(args=args)
    try:
        c1 = ArUcoTracker()
        c2 = ArUcoPublisher()
        
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