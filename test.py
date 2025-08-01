import numpy as np
import os
import cv2

# # path = "/home/yufeiyang/Documents/FoundationPose/output/ob_0000001/depth_enhanced/0000000.png"
path = "/home/yufeiyang/Documents/FoundationPose/output/ob_0000002/depth_enhanced/0000000.png"
# Load the depth image
depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
depth_image = np.array(depth_image, dtype=np.float32)
d = depth_image[depth_image < 1000]
np.set_printoptions(threshold=np.inf)
depth_image = np.clip(depth_image, 200, 1000)  # Clip values to a reasonable range
depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)
cv2.imshow("Depth Hotmap", depth_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

depth_path = "/home/yufeiyang/Documents/FoundationPose/output/ob_000/depth.png"
mask_path = "/home/yufeiyang/Documents/FoundationPose/output/ob_0000002/mask/0000000.png"
rgb_path = "/home/yufeiyang/Documents/FoundationPose/output/ob_0000002/rgb/0000000.png"

rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

depth_image = np.array(depth_image, dtype=np.float32)
mask_image = np.array(mask_image)
mask_image = mask_image.astype(bool)
# depth_image_masked = np.where(mask_image, depth_image, 0)  # Apply mask to depth image
depth_image_masked = cv2.bitwise_and(depth_image, depth_image, mask=mask_image.astype(np.uint8))


# depth_image_masked = cv2.applyColorMap(depth_image_masked.astype(np.uint8), cv2.COLORMAP_JET)
# cv2.imshow("Depth Hotmap", depth_image_masked)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

rgb_image = np.array(rgb_image, dtype=np.uint8)
rgb_image_masked = np.where(mask_image[..., None], rgb_image, 0)  # Apply mask to RGB image
rgb_image_masked = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_image.astype(np.uint8))
# rgb_image_masked = cv2.applyColorMap(rgb_image_masked, cv2.COLORMAP_JET)
cv2.imshow("RGB Masked", rgb_image_masked)
cv2.waitKey(0)  
