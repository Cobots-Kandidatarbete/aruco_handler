import numpy as np
import cv2
import transforms3d
import mediapipe as mp
import threading

import rclpy
import time
from rclpy.node import Node
from mp_msgs.msg import Landmark
from mp_msgs.msg import Hand
from mp_msgs.msg import Hands
from geometry_msgs.msg import Transform, TransformStamped
from builtin_interfaces.msg import Time
from tf2_msgs.msg import TFMessage
from rclpy.executors import MultiThreadedExecutor

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# camera_matrix:
#   cols: 3
#   data: [955.9252891407828, 0, 299.2929814576621, 0, 958.9317260791769, 193.5121531452791, 0, 0, 1]
#   rows: 3
# camera_name: camera1
# distortion_coefficients:
#   cols: 5
#   data: [0.09396209339360358, 0.6063644283226572, -0.02050973463465562, -0.01056330669804135, -4.163968515021963]
#   rows: 1
# distortion_model: plumb_bob
# focal_length_meters: 0
# image_height: 480
# image_width: 640
# projection_matrix:
#   cols: 4
#   data: [955.9252891407828, 0, 299.2929814576621, 0, 0, 958.9317260791769, 193.5121531452791, 0, 0, 0, 1, 0]
#   rows: 3
# rectification_matrix:
#   cols: 3
#   data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
#   rows: 3

# old c270 camera parameters
# calibration_matrix = np.array([
#         [1578.135315108312, 0.0, 625.6708621029746],
#         [0.0, 1585.223944490997, 274.1438454056999],
#         [0.0, 0.0, 1.0]
#     ]);

# new c270 camera parameters
calibration_matrix = np.array([
        [955.9252891407828, 0.0, 299.2929814576621],
        [0.0, 958.9317260791769, 193.5121531452791],
        [0.0, 0.0, 1.0]
    ]);

distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0])
# distortion_coefficients = np.array([0.09396209339360358, 0.6063644283226572, -0.02050973463465562, -0.01056330669804135, -4.163968515021963])
xi = np.array([0.0]) # what the hell is this? 
w, h = (640, 480)

rectification_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
projection_matrix = np.array([[955.9252891407828, 0.0, 299.2929814576621, 0.0], [0.0, 958.9317260791769, 193.5121531452791, 0.0], [0.0, 0.0, 1.0, 0.0]])
# projection_matrix = np.array([[1578.135315108312, 0.0, 0.0, 0.0], [0.0, 1585.223944490997, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

class Things:
    hands = []
    arucos = []

class Vision(Node):
    def __init__(self):
        super().__init__("vision")

        cv2.ShowUndistortedImage = True
        self.capture = cv2.VideoCapture(1)
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
        # new_cam_mtx, valid_roi = cv2.getOptimalNewCameraMatrix(calibration_matrix, distortion_coefficients, (w, h), 1, (w, h)) 
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.mapx, self.mapy = cv2.omnidir.initUndistortRectifyMap(
            calibration_matrix, 
            distortion_coefficients, 
            xi,
            rectification_matrix, 
            projection_matrix,
            # new_cam_mtx,
            (w,h), 
            cv2.CV_32F, 
            cv2.omnidir.RECTIFY_PERSPECTIVE
        )
   
        # mapx, mapy = cv2.initUndistortRectifyMap(
        #     calibration_matrix, 
        #     distortion_coefficients, 
        #     rectification_matrix,
        #     # new_cam_mtx,
        #     projection_matrix,
        #     (w,h),
        #     cv2.CV_32FC1
        # )

        # self.mapx, self.mapy = cv2.initUndistortRectifyMap(
        #     calibration_matrix, 
        #     distortion_coefficients, 
        #     rectification_matrix,
        #     calibration_matrix,
        #     # new_cam_mtx,
        #     (w,h),
        #     cv2.CV_32FC1
        # )

        self.get_logger().info("Vision Node should be started.")

        self.run_vision()

    def run_vision(self):
        def run_vision_callback_local():
            while True:
                # print("ASDFASDF")
                
                success, image = self.capture.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                
                with mp_hands.Hands(
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

                      # To improve performance, optionally mark the image as not writeable to
                      # pass by reference.
                        image.flags.writeable = False
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands.process(image)

                      # Draw the hand annotations on the image.
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        hands_msg = Hands()
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                single_hand_msg = Hand()
                                mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())

                                wrist_landmark = Landmark()
                                wrist_landmark.name = "wrist"
                                wrist_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].visibility
                                wrist_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w
                                wrist_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * w
                                wrist_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

                                thumb_cmc_landmark = Landmark()
                                thumb_cmc_landmark.name = "thumb_cmc"
                                thumb_cmc_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].visibility
                                thumb_cmc_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * w
                                thumb_cmc_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * w
                                thumb_cmc_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z

                                thumb_mcp_landmark = Landmark()
                                thumb_mcp_landmark.name = "thumb_mcp"
                                thumb_mcp_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].visibility
                                thumb_mcp_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * w
                                thumb_mcp_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * w
                                thumb_mcp_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z

                                thumb_ip_landmark = Landmark()
                                thumb_ip_landmark.name = "thumb_ip"
                                thumb_ip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].visibility
                                thumb_ip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * w
                                thumb_ip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * w
                                thumb_ip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z

                                thumb_tip_landmark = Landmark()
                                thumb_tip_landmark.name = "thumb_tip"
                                thumb_tip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].visibility
                                thumb_tip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w
                                thumb_tip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * w
                                thumb_tip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z

                                index_finger_mcp_landmark = Landmark()
                                index_finger_mcp_landmark.name = "index_finger_mcp"
                                index_finger_mcp_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].visibility
                                index_finger_mcp_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * w
                                index_finger_mcp_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * w
                                index_finger_mcp_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

                                index_finger_pip_landmark = Landmark()
                                index_finger_pip_landmark.name = "index_finger_pip"
                                index_finger_pip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].visibility
                                index_finger_pip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * w
                                index_finger_pip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * w
                                index_finger_pip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z

                                index_finger_dip_landmark = Landmark()
                                index_finger_dip_landmark.name = "index_finger_dip"
                                index_finger_dip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].visibility
                                index_finger_dip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * w
                                index_finger_dip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * w
                                index_finger_dip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z

                                index_finger_tip_landmark = Landmark()
                                index_finger_tip_landmark.name = "index_finger_tip"
                                index_finger_tip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].visibility
                                index_finger_tip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w
                                index_finger_tip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * w
                                index_finger_tip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

                                middle_finger_mcp_landmark = Landmark()
                                middle_finger_mcp_landmark.name = "middle_finger_mcp"
                                middle_finger_mcp_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].visibility
                                middle_finger_mcp_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * w
                                middle_finger_mcp_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * w
                                middle_finger_mcp_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z

                                middle_finger_pip_landmark = Landmark()
                                middle_finger_pip_landmark.name = "middle_finger_pip"
                                middle_finger_pip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].visibility
                                middle_finger_pip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * w
                                middle_finger_pip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * w
                                middle_finger_pip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z

                                middle_finger_dip_landmark = Landmark()
                                middle_finger_dip_landmark.name = "middle_finger_dip"
                                middle_finger_dip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].visibility
                                middle_finger_dip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * w
                                middle_finger_dip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * w
                                middle_finger_dip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z

                                middle_finger_tip_landmark = Landmark()
                                middle_finger_tip_landmark.name = "middle_finger_tip"
                                middle_finger_tip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].visibility
                                middle_finger_tip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * w
                                middle_finger_tip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * w
                                middle_finger_tip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z

                                ring_finger_mcp_landmark = Landmark()
                                ring_finger_mcp_landmark.name = "ring_finger_mcp"
                                ring_finger_mcp_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].visibility
                                ring_finger_mcp_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * w
                                ring_finger_mcp_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * w
                                ring_finger_mcp_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z

                                ring_finger_pip_landmark = Landmark()
                                ring_finger_pip_landmark.name = "ring_finger_pip"
                                ring_finger_pip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].visibility
                                ring_finger_pip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * w
                                ring_finger_pip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * w
                                ring_finger_pip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z

                                ring_finger_dip_landmark = Landmark()
                                ring_finger_dip_landmark.name = "ring_finger_dip"
                                ring_finger_dip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].visibility
                                ring_finger_dip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * w
                                ring_finger_dip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * w
                                ring_finger_dip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z

                                ring_finger_tip_landmark = Landmark()
                                ring_finger_tip_landmark.name = "ring_finger_tip"
                                ring_finger_tip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].visibility
                                ring_finger_tip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * w
                                ring_finger_tip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * w
                                ring_finger_tip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z

                                pinky_mcp_landmark = Landmark()
                                pinky_mcp_landmark.name = "pinky_mcp"
                                pinky_mcp_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].visibility
                                pinky_mcp_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * w
                                pinky_mcp_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * w
                                pinky_mcp_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z

                                pinky_pip_landmark = Landmark()
                                pinky_pip_landmark.name = "pinky_pip"
                                pinky_pip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].visibility
                                pinky_pip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * w
                                pinky_pip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * w
                                pinky_pip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z

                                pinky_dip_landmark = Landmark()
                                pinky_dip_landmark.name = "pinky_dip"
                                pinky_dip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].visibility
                                pinky_dip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * w
                                pinky_dip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * w
                                pinky_dip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z

                                pinky_tip_landmark = Landmark()
                                pinky_tip_landmark.name = "pinky_tip"
                                pinky_tip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].visibility
                                pinky_tip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * w
                                pinky_tip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * w
                                pinky_tip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z

                                single_hand_msg.wrist = wrist_landmark
                                single_hand_msg.thumb_cmc = thumb_cmc_landmark
                                single_hand_msg.thumb_mcp = thumb_mcp_landmark
                                single_hand_msg.thumb_ip = thumb_ip_landmark
                                single_hand_msg.index_finger_mcp = index_finger_mcp_landmark
                                single_hand_msg.index_finger_pip = index_finger_pip_landmark
                                single_hand_msg.index_finger_dip = index_finger_dip_landmark
                                single_hand_msg.index_finger_tip = index_finger_tip_landmark
                                single_hand_msg.middle_finger_mcp = middle_finger_mcp_landmark
                                single_hand_msg.middle_finger_pip = middle_finger_pip_landmark
                                single_hand_msg.middle_finger_dip = middle_finger_dip_landmark
                                single_hand_msg.middle_finger_tip = middle_finger_tip_landmark
                                single_hand_msg.ring_finger_mcp = ring_finger_mcp_landmark
                                single_hand_msg.ring_finger_pip = ring_finger_pip_landmark
                                single_hand_msg.ring_finger_dip = ring_finger_dip_landmark
                                single_hand_msg.ring_finger_tip = ring_finger_tip_landmark
                                single_hand_msg.pinky_mcp = pinky_mcp_landmark
                                single_hand_msg.pinky_pip = pinky_pip_landmark
                                single_hand_msg.pinky_dip = pinky_dip_landmark
                                single_hand_msg.pinky_tip = pinky_tip_landmark
          
                                Things.hands.append(single_hand_msg)

                frame = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)
                # corners, ids, rejected = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_params)
                corners, ids, rejected = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
                corners = np.array(corners)

                if len(corners) > 0:
                    # flatten the ArUco IDs list
                    ids = ids.flatten()
                    # loop over the detected ArUCo corners

                    for (markerCorner, markerID) in zip(corners, ids):
                        # extract the marker corners (which are always returned in
                        # top-left, top-right, bottom-right, and bottom-left order)
                        corners = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corners
                        # convert each of the (x, y)-coordinate pairs to integers
                        topRight = (int(topRight[0]), int(topRight[1]))
                        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                        topLeft = (int(topLeft[0]), int(topLeft[1]))
                        # draw the bounding box of the ArUCo detection
                        # cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                        # cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                        # cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                        # cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                        cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                        cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                        cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                        cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
                        # compute and draw the center (x, y)-coordinates of the ArUco
                        # marker
                        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                        cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                        # cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                        # draw the ArUco marker ID on the image
                        # cv2.putText(image, str(markerID),
                        cv2.putText(frame, str(markerID),
                                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                        # compute the pose
                        ret = cv2.aruco.estimatePoseSingleMarkers(markerCorner, 0.05, calibration_matrix, distortion_coefficients);
                        (rot, trans) = (ret[0][0, 0, :], ret[1][0, 0, :])

                        dst,jacobian = cv2.Rodrigues(rot)
                        quat = transforms3d.quaternions.mat2quat(dst)

                        # print(trans)

                        t = Transform()
                        t.translation.x = trans[0]
                        t.translation.y = trans[1]
                        t.translation.z = trans[2]

                        t.rotation.w = quat[0]
                        t.rotation.x = quat[1]
                        t.rotation.y = quat[2]
                        t.rotation.z = quat[3]

                        stamped = TransformStamped()
                        stamped.header.frame_id = "logitech_c270"
                        stamped.header.stamp = Time()
                        current_time = self.get_clock().now().seconds_nanoseconds()
                        stamped.header.stamp.sec = current_time[0]
                        stamped.header.stamp.nanosec = current_time[1]

                        stamped.child_frame_id = "aruco_" + str(markerID)
                        stamped.transform = t

                        Things.arucos.append(stamped)

                        # cv2.aruco.drawAxis(image, calibration_matrix, distortion_coefficients, rot, trans, 0.15)
                        cv2.aruco.drawAxis(frame, calibration_matrix, distortion_coefficients, rot, trans, 0.15)

                # cv2.imshow('img', image)
                cv2.imshow('img', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            self.capture.release() 
            cv2.destroyAllWindows()
            
        t1 = threading.Thread(target=run_vision_callback_local)
        t1.daemon = True
        t1.start()

class VisionHandler(Node):
    def __init__(self):
        super().__init__("vision_handler")

        self.tf_publisher = self.create_publisher(TFMessage, "/tf", 20)
        self.hands_publisher_ = self.create_publisher(Hands, "/hands", 20)
        self.timer = self.create_timer(0.1, self.hands_timer_callback)

        self.get_logger().info("Vision Handler Node should be started.")

    def hands_timer_callback(self):
        hands_msg = Hands()
        hands_msg.hands = Things.hands
        self.hands_publisher_.publish(hands_msg)
        Things.hands = []
        msgs = []

        for t in Things.arucos:
            msgs.append(t)
            
        tf_msg = TFMessage()
        tf_msg.transforms = msgs
        self.tf_publisher.publish(tf_msg)

        Things.arucos = []


def main(args=None):
    rclpy.init(args=args)
    try:
        c1 = VisionHandler()
        c2 = Vision()

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


