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


def tup_to_int(tup):
    x, y = tup
    return int(x), int(y)


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


class ArUcoTracker(Node):
    def __init__(self, camera_index=6, marker_size=0.05):
        super().__init__(node_name="aruco_tracker")

        aruco_dict=cv2.aruco.DICT_6X6_50
        cv2.ShowUndistortedImage = True
        self.marker_size = marker_size
        self.capture = cv2.VideoCapture(camera_index)
        self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        cam_name = "realsense_d435"
        self.cam_name = cam_name

        path_to_calibrator_dir = f"{os.path.curdir}/src/camera-calibrator/{cam_name}_configs"
        if not os.path.isdir(path_to_calibrator_dir):
            raise ImportError(f"Could not find dir {path_to_calibrator_dir}")

        path_to_calibration_file = f"{path_to_calibrator_dir}/{cam_name}_config.yaml"
        if not os.path.isfile(path_to_calibration_file):
            raise ImportError(f"Could not find file {path_to_calibration_file}")


        try:
            calibration_file = cv2.FileStorage(path_to_calibration_file, cv2.FILE_STORAGE_READ)
            self.camera_matrix = calibration_file.getNode('camera_matrix').mat()
            self.distortion_coeffecients = calibration_file.getNode('distortion_matrix').mat()
            self.optimal_camera_matrix = calibration_file.getNode('optimal_camera_matrix').mat()
            self.image_size = calibration_file.getNode('image_size').mat().astype(int).flatten()

        finally:
            calibration_file.release()
        

        rectification_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float')
        map_1_type = cv2.CV_32F


        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.distortion_coeffecients,
            rectification_matrix,
            self.camera_matrix,
            self.image_size,
            map_1_type
        )


        self.get_logger().info("aruco_tracker node should be started")
        t1 = threading.Thread(target=self.run_vision_callback)
        t1.daemon = True
        t1.start()
        
    def run_vision_callback(self):
        while True:
            
            capture_success, frame = self.capture.read()
            if not capture_success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)

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

        self.capture.release()
        cv2.destroyAllWindows()

    def buffer_marker(self, marker_id, transform):
        transform_stamped = self.create_transform_stamp(marker_id, transform)
        StampContainer.aruco_stamps.append(transform_stamped)

    def get_pose_vectors(self, marker_corner):
        # http://amroamroamro.github.io/mexopencv/matlab/cv.estimatePoseSingleMarkers.html
        pose_estimation = cv2.aruco.estimatePoseSingleMarkers(marker_corner, self.marker_size, self.camera_matrix, self.distortion_coeffecients)
                    
        rotation = pose_estimation[0][0, 0, :]
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        rotation_quaternion = transforms3d.quaternions.mat2quat(rotation_matrix)

        translation = pose_estimation[1][0, 0, :]

        transform = self.get_pose_transform(rotation_quaternion, translation)
        return rotation,translation,transform

    def create_transform_stamp(self, marker_id, transform):
        transform_stamped = TransformStamped()
        transform_stamped.header.frame_id = self.cam_name
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
        cv2.aruco.drawAxis(frame, self.camera_matrix, self.distortion_coeffecients, rotation, translation, DrawConfig.axis_length)


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