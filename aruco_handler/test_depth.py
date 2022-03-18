import cv2
import numpy as np
import pyrealsense2 as rs


pipeline = None


def init():
    global pipeline

    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)


def get_frame():
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        return False, None, None
        
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return True, depth_image, color_image


def stream_frames():
    while True:
        ret, depth_frame, color_frame = get_frame()

        pt_x, pt_y = pt = 400, 300


        cv2.circle(color_frame, pt, 4, (0, 0, 255))
        distance = depth_frame[pt_y, pt_x]
        print(distance)
        cv2.imshow("depth frame", depth_frame)
        cv2.imshow("color frame", color_frame)


        input_key = cv2.waitKey(1)
        if input_key == 27:
            break



def release():
    pipeline.stop()



def main():
    init()
    
    stream_frames()

    release()

if __name__ == "__main__":
    main()
