# this file takes reference images and feeds to the train nerf
'''
data
--view 1
 --cam_in_ob
 --depth_enhanced
 --mask
 --model
 --rgb
 --select_frames.yml
 --K.txt
--view 2
...
'''


import datetime
import pyrealsense2 as rs
from estimater import *
from datareader import *
from FoundationPose.mask import *
import argparse
from FoundationPose.lcm_systems.pose_publisher import PosePublisher
# imports for reading camera extrinsics
import yaml
import numpy as np
import os.path as op
from scipy.spatial.transform import Rotation as R
import time
import cv2

def process_depth(depth_image, mask):
    # Apply the mask to the depth image
    # masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)
    assert depth_image.shape == mask.shape, "Depth image and mask must have the same shape"
    # masked_depth = np.where(mask, depth_image, 0)
    masked_depth = depth_image * (mask > 0).astype(depth_image.dtype)
    return masked_depth

def get_cam_in_ob(pipeline, depth_frame, mask, center):
    # depth: masked depth image with only the object of interest
    # color: RGB image with only the object of interest

    x, y = center

    

    

def object_center(mask):
    ys, xs = np.where(mask)
    if len(xs) > 0:
        cs = int(np.mean(xs))
        rs = int(np.mean(ys))
    else:
        cs, rs = None, None
    return cs, rs

def main():

    print("Starting the camera stream...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # breakpoint()
    pipeline.start(config)
    print("Press Enter to capture a key frame, or 'q' to quit.")

    frame_idx = 0

    # Get camera instrinsics
    # Wait for a frame to get intrinsics
    frame0 = pipeline.wait_for_frames()
    color_frame = frame0.get_color_frame()

    # Get intrinsics from the color stream
    color_intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
    # breakpoint()
    # Build camera intrinsic matrix
    K = np.array([
        [color_intrinsics.fx, 0, color_intrinsics.ppx],
        [0, color_intrinsics.fy, color_intrinsics.ppy],
        [0, 0, 1]
    ])

    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            cv2.imshow("RGB Stream", color_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == 13:  # Enter key
                print(f"Key frame {frame_idx} captured. Select mask by clicking points, press Enter when done.")

                # Mask selection
                points = []
                image_display = color_image.copy()

                def select_points(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        points.append((x, y))
                        cv2.circle(image_display, (x, y), 3, (0, 255, 0), -1)
                        cv2.imshow("Select Mask", image_display)

                cv2.namedWindow("Select Mask")
                cv2.setMouseCallback("Select Mask", select_points)

                while True:
                    cv2.imshow("Select Mask", image_display)
                    mask_key = cv2.waitKey(1) & 0xFF
                    if mask_key == 13:  # Enter key
                        break

                mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
                if points:
                    points_array = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [points_array], 255)

                # Save images and mask
                os.makedirs("output", exist_ok=True)
                # Create a subfolder for each frame
                frame_folder = op.join("output", "ob_0000001")
                os.makedirs(frame_folder, exist_ok=True)

                
                rgb_path = op.join(frame_folder, "rgb", "0000000.png")
                depth_path = op.join(frame_folder, "depth_enhanced", "0000000.png")
                o_depth_path = op.join(frame_folder, "depth.png")
                mask_path = op.join(frame_folder, "mask", "0000000.png")

                cv2.imwrite(rgb_path, color_image)
                processed_depth = process_depth(depth_image, mask)
                center_x, center_y = object_center(mask)
                get_cam_in_ob(pipeline, depth_frame, mask, (center_x, center_y))

                #test center
                # mask_uint8 = (mask * 255).astype(np.uint8)
                # mask_color = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
                # cv2.circle(mask, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)
                ###########3


                # Normalize processed depth for saving as PNG
                cv2.imwrite(depth_path, processed_depth)
                cv2.imwrite(o_depth_path, depth_image)
                # cv2.imwrite(mask_path, mask)
                cv2.imwrite(mask_path, mask)
                print(f"Saved: {rgb_path}, {depth_path}, {mask_path}")
                # save k in a txt file
                K_path = op.join(frame_folder, "K.txt")
                np.savetxt(K_path, K, fmt='%.6f')
                print(f"Saved camera intrinsics to {K_path}")

                cv2.destroyWindow("Select Mask")
                frame_idx += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()