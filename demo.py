import argparse
import cv2
import keyboard
import numpy as np
import open3d as o3d
import os
import pygame
from transforms3d.axangles import axangle2mat

import config

from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from utils import OneEuroFilter, imresize
from wrappers import ModelPipeline
from utils import *

def video_to_images(vid_file, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)
    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder


def run(args):
    ############ output visualization ############
    # view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
    # window_size = 1080

    # hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    # mesh.vertices = \
    # o3d.utility.Vector3dVector(np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
    # mesh.compute_vertex_normals()

    # viewer = o3d.visualization.Visualizer()
    # viewer.create_window(
    # width=window_size + 1, height=window_size + 1,
    # window_name='Minimal Hand - output'
    # )
    # viewer.add_geometry(mesh)

    # view_control = viewer.get_view_control()
    # cam_params = view_control.convert_to_pinhole_camera_parameters()
    # extrinsic = cam_params.extrinsic.copy()
    # extrinsic[0:3, 3] = 0
    # cam_params.extrinsic = extrinsic
    # cam_params.intrinsic.set_intrinsics(
    # window_size + 1, window_size + 1, config.CAM_FX, config.CAM_FY,
    # window_size // 2, window_size // 2
    # )
    # view_control.convert_from_pinhole_camera_parameters(cam_params)
    # view_control.set_constant_z_far(1000)

    # render_option = viewer.get_render_option()
    # render_option.load_from_json('./render_option.json')
    # viewer.update_renderer()

    # ############ input visualization ############
    # pygame.init()
    # display = pygame.display.set_mode((window_size, window_size))
    # pygame.display.set_caption('Minimal Hand - input')

    # ############ misc ############
    # mesh_smoother = OneEuroFilter(4.0, 0.0)
    # clock = pygame.time.Clock()

    ############ Move all of above code to local to render ###########

    video_file = args.vid_file
    if not os.path.isfile(video_file):
        exit(f'Input video \"{video_file}\" does not exist!')

    output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('.mp4', ''))
    os.makedirs(output_path, exist_ok=True)

    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]

    # total_time = time.time()
    import pdb; pdb.set_trace()

    image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]

    model = ModelPipeline()

    for i in image_file_names:
        # What do all these conditions check for?
        frame_large = x
        if frame_large is None:
            continue
        if frame_large.shape[0] > frame_large.shape[1]:
            margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
            frame_large = frame_large[margin:-margin]
        else:
            margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
            frame_large = frame_large[:, margin:-margin]

        frame_large = np.flip(frame_large, axis=1).copy() # why? Camera flip?
        frame = imresize(frame_large, (128, 128))  # needed

        ######## Golden lines, run this here #########
        _, theta_mpii = model.process(frame) 
        theta_mano = mpii_to_mano(theta_mpii)

        ######## Save theta_mano and pass as input to local ######## 
        v = hand_mesh.set_abs_quat(theta_mano)
        v *= 2 # for better visualization
        v = v * 1000 + np.array([0, 0, 400])
        v = mesh_smoother.process(v)

        mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
        mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
        mesh.paint_uniform_color(config.HAND_COLOR)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        # for some version of open3d you may need `viewer.update_geometry(mesh)`
        viewer.update_geometry()

        viewer.poll_events()

        display.blit(
            pygame.surfarray.make_surface(
            np.transpose(
                imresize(frame_large, (window_size, window_size)
            ), (1, 0, 2))
            ),
            (0, 0)
        )
        pygame.display.update()

        if keyboard.is_pressed("esc"):
            break

        clock.tick(30) # What's this do? If it adds delay remove it


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_file', type=str,
                    help='input video path or youtube link')
                    
    args = parser.parse_args()
    run(args)
