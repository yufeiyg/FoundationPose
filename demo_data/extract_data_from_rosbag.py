import numpy as np
import rosbag
import imageio
from cv_bridge import CvBridge
import os.path as op
from tqdm import tqdm
import pdb

RGB_ROS_TOPIC = '/device_0/sensor_1/Color_0/image/data'
DEPTH_ROS_TOPIC = '/device_0/sensor_0/Depth_0/image/data'

TIME_EXCESS_BUFFER = (1/30.0)/2
TIME_SYNCHRONIZATION_TOLERANCE = TIME_EXCESS_BUFFER


def extract_synchronized_rgb_and_depth_images(
        bag_file: str, rgb_output_dir: str, depth_output_dir: str,
        dataset_dir: str
):
    depth_bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()

    # Extract the RGB and depth images from the depth bag, including all frames
    # within the specified start and end times.
    topics = [DEPTH_ROS_TOPIC, RGB_ROS_TOPIC]
    depth_images, rgb_images = {}, {}
    for (topic, msg, _t) in depth_bag.read_messages(topics=topics):
        if topic == DEPTH_ROS_TOPIC:
            cv_img_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")

            # If we want to smooth the depth map
            # cv_img_depth = cv2.bilateralFilter(cv_img_depth.astype(
            # np.float32), 20, 20, 10).astype(np.uint16)
            # # diameter, sigmaColor, sigmaSpace

            # Store the result.
            depth_images[msg.header.stamp.to_sec()] = cv_img_depth

        elif topic == RGB_ROS_TOPIC:
            cv_img_rgb = bridge.imgmsg_to_cv2(msg, "rgb8")
            rgb_images[msg.header.stamp.to_sec()] = cv_img_rgb

    # Synchronize the depth and RGB images based on the timestamps, writing the
    # synchronized results to the depth and RGB output directories.
    image_i = 1
    depth_times = []
    for depth_time in tqdm(depth_images, desc='Writing synchronized depth and RGB images'):
        closest_rgb_time = min(
            rgb_images.keys(),
            key=lambda t: abs(t - depth_time)
        )
        imageio.imwrite(
            op.join(depth_output_dir, f'{image_i:04d}.png'),
            np.uint16(depth_images[depth_time])
        )
        imageio.imwrite(
            op.join(rgb_output_dir, f'{image_i:04d}.png'),
            rgb_images[closest_rgb_time]
        )
        # print(f'Wrote synchronized depth and RGB images {image_i}')
        image_i += 1
        depth_times.append(depth_time)

    # Write the BundleSDF timestamps to file.
    np.savetxt(op.join(dataset_dir, 'depth_message_timestamps.txt'),
               depth_times)

    depth_bag.close()


extract_synchronized_rgb_and_depth_images(
    bag_file='/home/sharanya/workspace/foundationPose/20240625_162638.bag',
    rgb_output_dir='/home/sharanya/workspace/foundationPose/test_bag_output/rgb',
    depth_output_dir='/home/sharanya/workspace/foundationPose/test_bag_output/depth',
    dataset_dir='/home/sharanya/workspace/foundationPose/test_bag_output'
)
