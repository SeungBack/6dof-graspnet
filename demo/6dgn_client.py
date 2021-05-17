from __future__ import print_function

import sys
import numpy as np
import argparse
import grasp_estimator

import os
import tensorflow as tf
import glob
import mayavi.mlab as mlab
from visualization_utils import *
import mayavi.mlab as mlab

from easy_tcp_python2_3 import socket_utils as su

def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--vae_checkpoint_folder',
        type=str, 
        default='checkpoints/latent_size_2_ngpus_1_gan_1_confidence_weight_0.1_npoints_1024_num_grasps_per_object_256_train_evaluator_0_')
    parser.add_argument(
        '--evaluator_checkpoint_folder', 
        type=str, 
        default='checkpoints/npoints_1024_train_evaluator_1_allowed_categories__ngpus_8_/'
    )
    parser.add_argument(
        '--gradient_based_refinement',
        action='store_true',
        default=False,
    )
    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument('--threshold', type=float, default=0.8)

    return parser


def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y,x,:]
    
    return pc_colors


def backproject(depth_cv, intrinsic_matrix, return_finite_depth=True, return_selection=False):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection
        
    return X


def main(args):

    sock = su.initialize_client('localhost', 7777)

    parser = make_parser()
    args = parser.parse_args(args)
    cfg = grasp_estimator.joint_config(
        args.vae_checkpoint_folder,
        args.evaluator_checkpoint_folder,
    )
    cfg['threshold'] = args.threshold
    cfg['sample_based_improvement'] = 1 - int(args.gradient_based_refinement)
    cfg['num_refine_steps'] = 10 if args.gradient_based_refinement else 20
    estimator = grasp_estimator.GraspEstimator(cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    sess = tf.Session()
    estimator.build_network()
    estimator.load_weights(sess)

    while True:
        data = su.recvall_pickle(sock)

        print(data.keys())
        for k in data.keys():
            print(k, np.shape(data[k]))

        depth = data['depth']
        image = data['image']
        K = data['intrinsics_matrix']
        # Removing points that are farther than 1 meter or missing depth 
        # values.
        depth[depth == 0] = np.nan
        depth[depth > 1] = np.nan
        pc, selection = backproject(depth, K, return_finite_depth=True, return_selection=True)
        pc_colors = image.copy()
        pc_colors = np.reshape(pc_colors, [-1, 3])
        pc_colors = pc_colors[selection, ::-1]

        # down sampling
        idx = np.random.choice(pc_colors.shape[0], 100000, replace=False)
        pc = pc[idx, :]
        pc_colors = pc_colors[idx, :]

        # Smoothed pc comes from averaging the depth for 10 frames and removing
        # the pixels with jittery depth between those 10 frames.
        object_pc = data['smoothed_object_pc']
        latents = estimator.sample_latents()
        generated_grasps, generated_scores, _ = estimator.predict_grasps(
            sess,
            object_pc,
            latents,
            num_refine_steps=cfg.num_refine_steps,
        )
        print("====>", generated_grasps, generated_scores)
        mlab.figure(bgcolor=(1,1,1))
        if len(generated_grasps) != 0:
            draw_scene(
                pc,
                pc_color=pc_colors,
                grasps=generated_grasps,
                grasp_scores=generated_scores,
            )
            print('close the window to continue to next object . . .')
        mlab.show()
        su.sendall_pickle(sock, [generated_grasps, generated_scores])

if __name__ == "__main__" :

    main(sys.argv[1:])

