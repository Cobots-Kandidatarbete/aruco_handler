
# pip requirements
# rtsp
# opencv-python
# opencv-contrib-python
# transforms3d

import numpy as np
import cv2
import transforms3d

import rclpy
import time
from rclpy.node import Node

from geometry_msgs.msg import Transform, TransformStamped
from builtin_interfaces.msg import Time
from tf2_msgs.msg import TFMessage

def main():
    cap = cv2.VideoCapture(0)


    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
    aruco_params = cv2.aruco.DetectorParameters_create()


    mtx = np.array([
            [ 1.8870806859867380e+03, 0., 1.3437304789046996e+03,],
            [ 0., 1.8871101256728850e+03, 7.5612309014810398e+02, ],
            [0., 0., 1. ]
        ]);
    dist = np.array([ 1.0037838130574537e-03, -9.1061781286498524e-04,
                      7.6765697181571183e-05, 2.2310695460447523e-04 ])
    xi = np.array([1.0])

    w, h = (2688, 1512)

    newcameramtx = np.array([[w/4, 0, w/2], [0, h/2.25, h/2], [0, 0, 1]])
    mapx, mapy = cv2.omnidir.initUndistortRectifyMap(mtx, dist, xi, None, newcameramtx,
                                                     (w,h), cv2.CV_32F, cv2.omnidir.RECTIFY_PERSPECTIVE)

    rclpy.init()
    node = rclpy.create_node("aruco")
    publisher = node.create_publisher(TransformStamped, "/aruco", 20)
    tf_publisher = node.create_publisher(TFMessage, "/tf", 20)

    while True:
        success, frame = cap.read()
        if not success:
            continue

        new_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        # cv2.imwrite('pre.jpg', frame)
        # cv2.imwrite('post.jpg', new_frame)

        # gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(new_frame, aruco_dict, parameters=aruco_params)
        corners = np.array(corners)
        # print(corners)
        print(ids)

        image = new_frame
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            msgs = []
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
                    ret = cv2.aruco.estimatePoseSingleMarkers(markerCorner, 0.153, newcameramtx, dist);
                    (rot, trans) = (ret[0][0, 0, :], ret[1][0, 0, :])

                    dst,jacobian = cv2.Rodrigues(rot)
                    quat = transforms3d.quaternions.mat2quat(dst)

                    t = Transform()
                    t.translation.x = trans[0]
                    t.translation.y = trans[1]
                    t.translation.z = trans[2]

                    t.rotation.w = quat[0]
                    t.rotation.x = quat[1]
                    t.rotation.y = quat[2]
                    t.rotation.z = quat[3]

                    stamped = TransformStamped()
                    stamped.header.frame_id = "ceiling_camera"
                    stamped.header.stamp = Time()
                    current_time = node.get_clock().now().seconds_nanoseconds()
                    stamped.header.stamp.sec = current_time[0]
                    stamped.header.stamp.nanosec = current_time[1]

                    stamped.child_frame_id = "aruco_" + str(markerID)
                    stamped.transform = t

                    publisher.publish(stamped)

                    msgs.append(stamped)

                    cv2.aruco.drawAxis(image, newcameramtx, dist, rot, trans, 0.15)


            tf_msg = TFMessage()
            tf_msg.transforms = msgs
            tf_publisher.publish(tf_msg)

            rclpy.spin_once(node, timeout_sec=0.1)
            #cv2.imwrite('markers.jpg', image)

    client.close()
    node.destroy()
