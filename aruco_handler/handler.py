# pip requirements
# rtsp
# opencv-python
# opencv-contrib-python
# transforms3d

import numpy as np
import cv2
import transforms3d
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from mp_msgs.msg import Landmark
from mp_msgs.msg import Hand
from mp_msgs.msg import Hands


import rclpy
import time
from rclpy.node import Node

from geometry_msgs.msg import Transform, TransformStamped
from builtin_interfaces.msg import Time
from tf2_msgs.msg import TFMessage


# c270 camera parameters
calibration_matrix = np.array([
        [1578.135315108312, 0.0, 625.6708621029746],
        [0.0, 1585.223944490997, 274.1438454056999],
        [0.0, 0.0, 1.0]
    ]);

distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
# distortion_coefficients = np.array([0.1913558390363024, 1.611580485047983, -0.0275432638538428, -0.0001706687576881858, -11.90379741245398])
xi = np.array([0.0]) # what the hell is this? 
w, h = (640, 480)

rectification_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
projection_matrix = np.array([[1578.135315108312, 0.0, 625.6708621029746, 0.0], [0.0, 1585.223944490997, 274.1438454056999, 0.0], [0.0, 0.0, 1.0, 0.0]])
# projection_matrix = np.array([[1578.135315108312, 0.0, 0.0, 0.0], [0.0, 1585.223944490997, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

def main():
    rclpy.init()
    node = rclpy.create_node("vision")
    aruco_publisher = node.create_publisher(TransformStamped, "/aruco", 20)
    hands_publisher = node.create_publisher(Hands, "/hands", 20)
    # tf_publisher = node.create_publisher(TFMessage, "/tf", 20)
    cv2.ShowUndistortedImage = True
    capture = cv2.VideoCapture(0)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
    # new_cam_mtx, valid_roi = cv2.getOptimalNewCameraMatrix(calibration_matrix, distortion_coefficients, (w, h), 1, (w, h)) 
    aruco_params = cv2.aruco.DetectorParameters_create()

    

    # mapx, mapy = cv2.omnidir.initUndistortRectifyMap(
    #     calibration_matrix, 
    #     distortion_coefficients, 
    #     xi,
    #     rectification_matrix, 
    #     projection_matrix,
    #     # new_cam_mtx,
    #     (w,h), 
    #     cv2.CV_32F, 
    #     cv2.omnidir.RECTIFY_PERSPECTIVE
    # )
   
    # mapx, mapy = cv2.initUndistortRectifyMap(
    #     calibration_matrix, 
    #     distortion_coefficients, 
    #     rectification_matrix,
    #     # new_cam_mtx,
    #     projection_matrix,
    #     (w,h),
    #     cv2.CV_32FC1
    # )

    mapx, mapy = cv2.initUndistortRectifyMap(
        calibration_matrix, 
        distortion_coefficients, 
        rectification_matrix,
        calibration_matrix,
        # new_cam_mtx,
        (w,h),
        cv2.CV_32FC1
    )

    while True:
        success, image = capture.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            # while cap.isOpened():
                
                
                if not success:
                    print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                    continue

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

                        thumb_mpc_landmark = Landmark()
                        thumb_mpc_landmark.name = "thumb_mpc"
                        thumb_mpc_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MPC].visibility
                        thumb_mpc_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MPC].x * w
                        thumb_mpc_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MPC].y * w
                        thumb_mpc_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MPC].z

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
                        index_finger_pip_landmark.visibility = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].visibility
                        index_finger_pip_landmark.x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * w
                        index_finger_pip_landmark.y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * w
                        index_finger_pip_landmark.z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z
                        
                        single_hand_msg.wrist = wrist_landmark
                        single_hand_msg.thumb_cmc = thumb_cmc_landmark
                        single_hand_msg.thumb_mpc = thumb_mpc_landmark
                        single_hand_msg.thumb_ip = thumb_ip_landmark
                        single_hand_msg.thumb_tip = thumb_tip_landmark
                        single_hand_msg.index_finger_mcp = index_finger_mcp_landmark
                        hands_msg.hands.append(single_hand_msg)

                hands_publisher.publish(hands_msg)

        frame = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
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
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

                # compute the pose
                ret = cv2.aruco.estimatePoseSingleMarkers(markerCorner, 0.05, calibration_matrix, distortion_coefficients);
                (rot, trans) = (ret[0][0, 0, :], ret[1][0, 0, :])

                dst,jacobian = cv2.Rodrigues(rot)
                quat = transforms3d.quaternions.mat2quat(dst)

                # t = Transform()
                # t.translation.x = trans[0]
                # t.translation.y = trans[1]
                # t.translation.z = trans[2]

                # t.rotation.w = quat[0]
                # t.rotation.x = quat[1]
                # t.rotation.y = quat[2]
                # t.rotation.z = quat[3]

                print(trans)

                cv2.aruco.drawAxis(image, calibration_matrix, distortion_coefficients, rot, trans, 0.15)

        rclpy.spin_once(node, timeout_sec=0.1)
        cv2.imshow('img', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    capture.release() 
    cv2.destroyAllWindows()
    node.destroy()

if __name__ == '__main__':
    main()

        
#         # this was working
#         # new_frame = image # cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
#         # cv2.imwrite('pre.jpg', frame)
#         # cv2.imwrite('post.jpg', new_frame)

#         # gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
#         corners, ids, rejected = cv2.aruco.detectMarkers(new_frame, aruco_dict, parameters=aruco_params)
#         corners = np.array(corners)
    # ret = cv2.aruco.estimatePoseSingleMarkers(markerCorner, 0.153, mtx, dist);
#                     (rot, trans) = (ret[0][0, 0, :], ret[1][0, 0, :])

#                     dst,jacobian = cv2.Rodrigues(rot)
#                     quat = transforms3d.quaternions.mat2quat(dst)

    # pass


















# def main():
#     cap = cv2.VideoCapture(0)

#     aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
#     aruco_params = cv2.aruco.DetectorParameters_create()

#     

#     w, h = (1280, 720)

# # cv::omnidir::initUndistortRectifyMap(K1, D1, xi1, R, P, s, CV_32FC1, Mapx, Mapy, cv::omnidir::RECTIFY_PERSPECTIVE);// Knew, new_size);

#     rclpy.init()
#     node = rclpy.create_node("aruco")
#     publisher = node.create_publisher(TransformStamped, "/aruco", 20)
#     tf_publisher = node.create_publisher(TFMessage, "/tf", 20)


#     # mtx = np.array([
#     #         [ 1.8870806859867380e+03, 0., 1.3437304789046996e+03,],
#     #         [ 0., 1.8871101256728850e+03, 7.5612309014810398e+02, ],
#     #         [0., 0., 1. ]
#     #     ]);
#     # dist = np.array([ 1.0037838130574537e-03, -9.1061781286498524e-04,
#     #                   7.6765697181571183e-05, 2.2310695460447523e-04 ])
#     # xi = np.array([1.0])

#     # w, h = (2688, 1512)

#     # newcameramtx = np.array([[w/4, 0, w/2], [0, h/2.25, h/2], [0, 0, 1]]) #whyyyyyyyyyyyyyyyyy?
#     mapx, mapy = cv2.omnidir.initUndistortRectifyMap(mtx, dist, xi, rectify, projection,
#                                                      (w,h), cv2.CV_32F, cv2.omnidir.RECTIFY_PERSPECTIVE)

    

#     while True:

#         success, image = cap.read()

#         with mp_hands.Hands(
#             model_complexity=0,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5) as hands:
#             # while cap.isOpened():
                
                
#                 if not success:
#                     print("Ignoring empty camera frame.")
#                 # If loading a video, use 'break' instead of 'continue'.
#                     continue

#               # To improve performance, optionally mark the image as not writeable to
#               # pass by reference.
#                 image.flags.writeable = False
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 results = hands.process(image)

#               # Draw the hand annotations on the image.
#                 image.flags.writeable = True
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#                 if results.multi_hand_landmarks:
#                     for hand_landmarks in results.multi_hand_landmarks:
#                         mp_drawing.draw_landmarks(
#                         image,
#                         hand_landmarks,
#                         mp_hands.HAND_CONNECTIONS,
#                         mp_drawing_styles.get_default_hand_landmarks_style(),
#                         mp_drawing_styles.get_default_hand_connections_style())
#                 # Flip the image horizontally for a selfie-view display.
#                 # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
#                 # if cv2.waitKey(5) & 0xFF == 27:
#                     # break

#         new_frame = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        
#         # this was working
#         # new_frame = image # cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
#         # cv2.imwrite('pre.jpg', frame)
#         # cv2.imwrite('post.jpg', new_frame)

#         # gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
#         corners, ids, rejected = cv2.aruco.detectMarkers(new_frame, aruco_dict, parameters=aruco_params)
#         corners = np.array(corners)
#         # print(corners)
#         print(ids)

#         image = new_frame
#         msgs = []
#         # stamped = TransformStamped()
#         # stamped.header.frame_id = "world"
#         # stamped.header.stamp = Time()
#         # current_time = node.get_clock().now().seconds_nanoseconds()
#         # stamped.header.stamp.sec = current_time[0]
#         # stamped.header.stamp.nanosec = current_time[1]

#         # stamped.child_frame_id = "logitech_c270"
#         # msgs.append(stamped)
        
#         if len(corners) > 0:
#             # flatten the ArUco IDs list
#             ids = ids.flatten()
#             # loop over the detected ArUCo corners

#             for (markerCorner, markerID) in zip(corners, ids):
#                     # extract the marker corners (which are always returned in
#                     # top-left, top-right, bottom-right, and bottom-left order)
#                     corners = markerCorner.reshape((4, 2))
#                     (topLeft, topRight, bottomRight, bottomLeft) = corners
#                     # convert each of the (x, y)-coordinate pairs to integers
#                     topRight = (int(topRight[0]), int(topRight[1]))
#                     bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
#                     bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
#                     topLeft = (int(topLeft[0]), int(topLeft[1]))
#                     # draw the bounding box of the ArUCo detection
#                     cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
#                     cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
#                     cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
#                     cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
#                     # compute and draw the center (x, y)-coordinates of the ArUco
#                     # marker
#                     cX = int((topLeft[0] + bottomRight[0]) / 2.0)
#                     cY = int((topLeft[1] + bottomRight[1]) / 2.0)
#                     cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
#                     # draw the ArUco marker ID on the image
#                     cv2.putText(image, str(markerID),
#                             (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
#                             0.5, (0, 255, 0), 2)

#                     # compute the pose
#                     ret = cv2.aruco.estimatePoseSingleMarkers(markerCorner, 0.153, mtx, dist);
#                     (rot, trans) = (ret[0][0, 0, :], ret[1][0, 0, :])

#                     dst,jacobian = cv2.Rodrigues(rot)
#                     quat = transforms3d.quaternions.mat2quat(dst)

#                     t = Transform()
#                     t.translation.x = trans[0]
#                     t.translation.y = trans[1]
#                     t.translation.z = trans[2]

#                     t.rotation.w = quat[0]
#                     t.rotation.x = quat[1]
#                     t.rotation.y = quat[2]
#                     t.rotation.z = quat[3]

#                     stamped = TransformStamped()
#                     stamped.header.frame_id = "logitech_c270"
#                     stamped.header.stamp = Time()
#                     current_time = node.get_clock().now().seconds_nanoseconds()
#                     stamped.header.stamp.sec = current_time[0]
#                     stamped.header.stamp.nanosec = current_time[1]

#                     stamped.child_frame_id = "aruco_" + str(markerID)
#                     stamped.transform = t

#                     publisher.publish(stamped)

#                     msgs.append(stamped)

#                     cv2.aruco.drawAxis(image, mtx, dist, rot, trans, 0.15)


#         tf_msg = TFMessage()
#         tf_msg.transforms = msgs
#         tf_publisher.publish(tf_msg)

#         rclpy.spin_once(node, timeout_sec=0.1)
#         cv2.imshow('img', image)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break

#     cap.release() 
#     cv2.destroyAllWindows()  
#     node.destroy()
