# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import pyrealsense2 as rs
from estimater import *
from datareader import *
from FoundationPose.mask import *
import argparse
from FoundationPose.lcm_systems.pose_publisher import PosePublisher
# imports for reading camera extrinsics
import yaml
from scipy.spatial.transform import Rotation as R


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/colored_jacktoy_data/mesh/jack_colored.obj')
  # parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  # parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/colored_jacktoy_data')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  print("This is the mesh file: " + args.mesh_file)
  mesh = trimesh.load(args.mesh_file,force='mesh')
  print("LOADED MESH FILE")

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  create_mask()
  mask = cv2.imread('mask.png')

  # Create a pipeline
  pipeline = rs.pipeline()

  # Create a config and configure the pipeline to stream
  config = rs.config()

  # Get device product line for setting a supporting resolution
  pipeline_wrapper = rs.pipeline_wrapper(pipeline)
  pipeline_profile = config.resolve(pipeline_wrapper)
  device = pipeline_profile.get_device()
  device_product_line = str(device.get_info(rs.camera_info.product_line))

  found_rgb = False
  for s in device.sensors:
      if s.get_info(rs.camera_info.name) == 'RGB Camera':
          found_rgb = True
          break
  if not found_rgb:
      print("The demo requires Depth camera with Color sensor")
      exit(0)

  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
  config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

  # Start streaming
  profile = pipeline.start(config)

  # Getting the depth sensor's depth scale (see rs-align example for explanation)
  depth_sensor = profile.get_device().first_depth_sensor()
  depth_scale = depth_sensor.get_depth_scale()
  print("Depth Scale is: " , depth_scale)

  # We will be removing the background of objects more than
  #  clipping_distance_in_meters meters away
  clipping_distance_in_meters = 1 #1 meter
  clipping_distance = clipping_distance_in_meters / depth_scale

  # Create an align object
  # rs.align allows us to perform alignment of depth frames to others frames
  # The "align_to" is the stream type to which we plan to align depth frames.
  align_to = rs.stream.color
  align = rs.align(align_to)

  i = 0
  # Make sure to update this value according to the current intrinsics from the 
  # camera. ros2 topic echo /camera/aligned_depth/camera_info from host machine.
  cam_K = np.array([[381.8276672363281, 0.0, 320.3140869140625],
                    [0.0, 381.4604187011719, 244.2602081298828],
                    [0.0, 0.0, 1.0]])
  
  # read camera extrinsics from the extrinsics.yaml file
  with open('extrinsics.yaml') as file:
      data_loaded = yaml.safe_load(file)
  cam_position_x = data_loaded['cam0']['pose']['position']['x']
  cam_position_y = data_loaded['cam0']['pose']['position']['y']
  cam_position_z = data_loaded['cam0']['pose']['position']['z']
  cam_orientation_x = data_loaded['cam0']['pose']['rotation']['x']
  cam_orientation_y = data_loaded['cam0']['pose']['rotation']['y']
  cam_orientation_z = data_loaded['cam0']['pose']['rotation']['z']

  cam_rotation = R.from_rotvec([cam_orientation_x, cam_orientation_y, cam_orientation_z])
  rotation_matrix = cam_rotation.as_matrix()
  world_to_cam = np.array([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], cam_position_x],
                      [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], cam_position_y],
                      [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], cam_position_z],
                      [0, 0, 0, 1]])

  ################## HERE ##################
Estimating = True
time.sleep(3)
# Streaming loop
try:
    while Estimating:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3
        color_image = np.asanyarray(color_frame.get_data())
    
        # Scale depth image to mm
        depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)

        # cv2.imshow('color', color_image)
        # cv2.imshow('depth', depth_image)
        
        if cv2.waitKey(1) == 13:
            Estimating = False
            break   
        
        logging.info(f'i:{i}')
        
        
        H, W = cv2.resize(color_image, (640,480)).shape[:2]
        color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
        
        depth[(depth<0.1) | (depth>=np.inf)] = 0
        
        if i==0:
            if len(mask.shape)==3:
                for c in range(3):
                    if mask[...,c].sum()>0:
                        mask = mask[...,c]
                        break
            mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
        
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            
            if debug>=3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, cam_K)
                valid = depth>=0.1
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
            
        else:
            pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)

        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{i}.txt', pose.reshape(4,4))

        if debug>=1:
            # center_pose = pose@np.linalg.inv(to_origin)
            cam_to_object = pose
            print("pose size: ", pose.shape)
            obj_pose_in_world = world_to_cam @ cam_to_object

            print("This is the object pose" + str(obj_pose_in_world))
            lcm_pose = PosePublisher()
            lcm_pose.publish_pose("Jack", obj_pose_in_world)
            vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=cam_to_object, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=cam_to_object, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
            cv2.waitKey(1)


        if debug>=2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{i}.png', vis)
        
        i += 1
            
            
        
finally:
    pipeline.stop()

