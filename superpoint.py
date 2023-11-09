#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/19 18:26
# @Author  : Wenjia
# @File    : homograph.py.py
# @Software: PyCharm

import cv2
import numpy
import numpy as np
import re
from matplotlib import pyplot as plt
import math
import os
import pandas as pd
import torch
import logging
import importlib
import yaml
from util import flattenDetection, nms_fast
from map_builder import initPatch, searchSpace, clipSpecificPatch

def rad(d):
    return d * math.pi / 180
def get_model(name):
    mod = __import__('models.{}'.format(name), fromlist=[''])
    return getattr(mod, name)

def modelLoader(model='SuperPointNet', **options):
    # create model
    logging.info("=> creating model: %s", model)
    net = get_model(model)
    net = net(**options)
    return net

def get_module(name):
    mod = importlib.import_module(name)
    return getattr(mod, name)

class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc

def get_pts_and_des(model, img, device):
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        outs = model(img.unsqueeze(0), 'SuperPoint')
        semi, coarse_desc = outs['semi'], outs['desc']

        heatmap = flattenDetection(semi, tensor=True)

        heatmap_np = heatmap.detach().cpu().numpy()

        pts_nms_batch = [getPtsFromHeatmap(h, conf_thresh=0.015, nms_dist=4) for h in heatmap_np]  # 0.015 4
        ## subpixel
        pts = soft_argmax_points(heatmap_np, pts_nms_batch, patch_size=5)

        desc_sparse = [sample_desc_from_points(outs['desc'], pts, device) for pts in pts_nms_batch]

        outs = {"pts": pts[0], "desc": desc_sparse[0]}
        return outs



def soft_argmax_points(heatmap, pts, patch_size=5):
    """
    input:
        pts: tensor [N x 2]
    """
    from util import toNumpy
    from util  import extract_patch_from_points
    from util  import soft_argmax_2d
    from util  import norm_patches

    ##### check not take care of batch #####
    # print("not take care of batch! only take first element!")
    pts = pts[0].transpose().copy()
    patches = extract_patch_from_points(heatmap, pts, patch_size=patch_size)
    import torch
    patches = np.stack(patches)
    patches_torch = torch.tensor(patches, dtype=torch.float32).unsqueeze(0)

    # norm patches
    patches_torch = norm_patches(patches_torch)

    from util import do_log
    patches_torch = do_log(patches_torch)
    # patches_torch = do_log(patches_torch)
    # print("one tims of log!")
    # print("patches: ", patches_torch.shape)
    # print("pts: ", pts.shape)

    dxdy = soft_argmax_2d(patches_torch, normalized_coordinates=False)
    # print("dxdy: ", dxdy.shape)
    points = pts
    points[:, :2] = points[:, :2] + dxdy.numpy().squeeze() - patch_size // 2
    patches = patches_torch.numpy().squeeze()
    pts_subpixel = [points.transpose().copy()]
    return pts_subpixel.copy()

def sample_desc_from_points(coarse_desc, pts, device):
    # --- Process descriptor.
    cell = 8
    H, W = coarse_desc.shape[2] * cell, coarse_desc.shape[3] * cell
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = np.zeros((D, 0))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts = torch.from_numpy(pts[:2, :].copy())
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        samp_pts = samp_pts.to(device)
        desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
        desc = desc.data.cpu().numpy().reshape(D, -1)
        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return desc

def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist):
    '''
    :param self:
    :param heatmap:
        np (H, W)
    :return:
    '''
    heatmap = heatmap.squeeze()
    border_remove = 4

    H, W = heatmap.shape[0], heatmap.shape[1]

    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    sparsemap = (heatmap >= conf_thresh)
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts

def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
            for k, v in data.items()}



def GetDistance(lon1, lat1, lon2, lat2):
    EARTH_RADIUS = 6371000
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lon1) - rad(lon2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * EARTH_RADIUS
    s = round(s * 10000) / 10000
    return s

def localization(test_ds, prediction):
    print('==> Loading pre-trained network.')
    # This class runs the SuperPoint network and processes its outputs.
    fe = SuperPointFrontend(weights_path='/home/lhl/data/superpoint/superpoint_kitti.pth',
                            nms_dist=4,
                            conf_thresh=0.015,
                            nn_thresh=0.7,
                            cuda='store_true')

    print('==> Successfully loaded pre-trained network.')

    MIN_MATCH_COUNT = 10

    Map_w = 4800
    Map_h = 2861
    lat1, lon1 = 47.05919722, 8.39368611  # 'bottom_left' E:454022.539, N:5211931.216
    lat2, lon2 = 47.07056667, 8.42165833  # 'top_right'   E:455978.326, N:5213187.329


    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    dataset_folder = test_ds.dataset_folder
    frame_path = dataset_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    error = []
    for i in range(test_ds.queries_num):
        img1_path = frame_path + df['file_name'][i]
        img1 = cv2.imread(img1_path, 0)  # query image
        img1 = cv2.resize(img1, (533, 400), interpolation=cv2.INTER_NEAREST)
        print("###################################{}#############################".format(i))
        best_match = 0
        for j in range(5):

            index = prediction[i][j]
            img2_path = test_ds.database_paths[index]

            img2 = cv2.imread(img2_path, 0)  # reference image

            pts1, desc1, heatmap1 = fe.run(img1.astype('float32')/ 255.)
            pts2, desc2, heatmap2 = fe.run(img2.astype('float32')/ 255.)

            # # find the keypoints and descriptiors with SIFT
            # kp1, des1 = sift.detectAndCompute(img1, None) # KeyPoint包含6个子项，pt, angle, response, size, octave, class_id：
            # kp2, des2 = sift.detectAndCompute(img2, None)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(desc1.T, desc2.T, k=2) # Finds the k best matches for each descriptor from a query set.
            print('matches:', len(matches), matches[0])
            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.9 * n.distance:
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:

                best_match = len(good)
                # print(kp1[805].pt) # pixel coordinate
                # find the pixel coordinate of the matched query and template points
                src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                # src_pts = np.float32([pts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                # dst_pts = np.float32([pts1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)


                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                print("lon_pre, lat_pre:", lon_pre, lat_pre)

                lat_gt, lon_gt = df['lat'][i], df['lon'][i]
                # E_gt, N_gt = 455213.39, 5212781.07
                print("lon_gt, lat_gt:", lon_gt, lat_gt)
                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)


                # cv2.drawMarker(img1, (int(x1), int(y1)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img1.jpg', img1)
                # cv2.drawMarker(img2, (int(trans_pt[0]), int(trans_pt[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img2.jpg', img2)
                #
                # draw_params = dict(
                #     matchColor=(0, 255, 0),
                #     singlePointColor=None,
                #     matchesMask=matchesMask,
                #     flags=2)
                # img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
                # img1_name = (re.findall('\d+', img1_path))[0]
                # text = str(img1_name) + "  and  " + str(patch_h) + '_' + str(patch_w) + '   matches:' + str(len(matches)) \
                #        + 'good:' + str(len(good)) + '   RMSE:   ' + str(RMSE)
                # plt.text(0, -1, text)
                # plt.imshow(img3)
                # file_name = str(img1_name) + '_' + str(patch_h) + '-' + str(patch_w) + '_' + str(j)
                # plt.savefig('./result/rule/{}'.format(file_name))

                # plt.show()


                # RMSE_UTM = np.sqrt(np.sum(np.square(E_gt-E_pre) + np.square(N_gt-N_pre)))
                # map = cv2.imread('/Users/lihaoling/Desktop/drive-download-20230512T090711Z-001/village/map_village.jpg', -1)
                # cv2.drawMarker(map, (int(trans_pt[0]+2464), int(trans_pt[1]+672)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=300, thickness=3, line_type=cv2.LINE_AA)
                # cv2.imwrite('village_map.jpg', map)

                print(RMSE)
        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)



def localization_village(args, test_ds, model, prediction):
    device = args.device

    MIN_MATCH_COUNT = 10

    # # dataset village
    Map_w = 4800
    Map_h = 2861
    lat1, lon1 = 47.05919722, 8.39368611  # 'bottom_left' E:454022.539, N:5211931.216
    lat2, lon2 = 47.07056667, 8.42165833  # 'top_right'   E:455978.326, N:5213187.329


    FLANN_INDEX_KDTREE = 1

    dataset_folder = test_ds.dataset_folder
    frame_path = dataset_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    error = []

    for i in range(test_ds.queries_num):
        img1_path = frame_path + df['file_name'][i]
        img1 = cv2.imread(img1_path, 0)  # query image
        img1 = cv2.resize(img1, (533, 400), interpolation=cv2.INTER_NEAREST) # village

        # img1 = cv2.resize(img1, (800, 600), interpolation=cv2.INTER_NEAREST) # gravel_pit
        image1 = img1
        img1 = img1.astype('float32') / 255.0
        outs1 = get_pts_and_des(model, img1 ,device=device)

        pts1, desc1 = outs1["pts"], outs1["desc"]
        print("###################################{}#############################".format(i))
        best_match = 0
        for j in range(20):

            index = prediction[i][j]
            img2_path = test_ds.database_paths[index]

            img2 = cv2.imread(img2_path, 0)
            image2 = img2# reference image
            img2 = img2.astype('float32') / 255.0
            outs2 = get_pts_and_des(model, img2, device=device)
            pts2, desc2 = outs2["pts"], outs2["desc"]



            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1.T, desc2.T, k=2) # village

            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.9 * n.distance:
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT:

                best_match = len(good)
                # print(kp1[805].pt) # pixel coordinate
                # find the pixel coordinate of the matched query and template points
                src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                # src_pts = np.float32([pts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                # dst_pts = np.float32([pts1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)


                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                # print("lon_pre, lat_pre:", lon_pre, lat_pre)

                lat_gt, lon_gt = df['lat'][i], df['lon'][i]

                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)


                # cv2.drawMarker(image1, (int(x1), int(y1)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img1.jpg', image1)
                # cv2.drawMarker(image2, (int(trans_pt[0]), int(trans_pt[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img2.jpg', image2)
                #
                # # kp1 = pcv2.KeyPoint(x=pts1,y=2,size=3)
                # # kp1 = cv2.Keypoint(numpy.delete(pts1, 2, axis=0))
                # kpts1 = numpy.delete(pts1, 2, axis=0)
                # kpts1 =kpts1.T
                # cv_kpts1 = [cv2.KeyPoint(kpts1[i][0], kpts1[i][1], 1)
                #             for i in range(kpts1.shape[0])]
                #
                # kpts2 = numpy.delete(pts2, 2, axis=0)
                # kpts2 = kpts2.T
                # cv_kpts2 = [cv2.KeyPoint(kpts2[i][0], kpts2[i][1], 1)
                #             for i in range(kpts2.shape[0])]
                #
                #
                #
                # draw_params = dict(
                #     matchColor=(0, 255, 0),
                #     singlePointColor=None,
                #     matchesMask=matchesMask,
                #     flags=2)
                # img3 = cv2.drawMatches(image1, cv_kpts1, image2, cv_kpts2, good, None, **draw_params)
                # img1_name = (re.findall('\d+', img1_path))[0]
                # text = str(img1_name) + "  and  " + str(patch_h) + '_' + str(patch_w) + '   matches:' + str(len(matches)) \
                #        + 'good:' + str(len(good)) + '   RMSE:   ' + str(RMSE)
                # plt.text(0, -1, text)
                # plt.imshow(img3)
                # # file_name = str(img1_name) + '_' + str(patch_h) + '-' + str(patch_w) + '_' + str(j)
                # # plt.savefig('./result/rule/{}'.format(file_name))
                #
                # plt.show()

                print(RMSE)
        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)

def localization_village_SIFT(args, test_ds, model, prediction):
    device = args.device

    MIN_MATCH_COUNT = 10

    # # dataset village
    Map_w = 4800
    Map_h = 2861
    lat1, lon1 = 47.05919722, 8.39368611  # 'bottom_left' E:454022.539, N:5211931.216
    lat2, lon2 = 47.07056667, 8.42165833  # 'top_right'   E:455978.326, N:5213187.329


    FLANN_INDEX_KDTREE = 1

    dataset_folder = test_ds.dataset_folder
    frame_path = dataset_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    error = []

    for i in range(test_ds.queries_num):
        img1_path = frame_path + df['file_name'][i]
        img1 = cv2.imread(img1_path, 0)  # query image
        img1 = cv2.resize(img1, (533, 400), interpolation=cv2.INTER_NEAREST) # village

        sift = cv2.SIFT_create()
        pts1, desc1 = sift.detectAndCompute(img1, None)


        print("###################################{}#############################".format(i))
        best_match = 0
        for j in range(100):

            index = prediction[i][j]
            img2_path = test_ds.database_paths[index]

            img2 = cv2.imread(img2_path, 0)

            pts2, desc2 = sift.detectAndCompute(img2, None)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)  # Finds the k best matches for each descriptor from a query set.


            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.8 * n.distance:
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT:

                best_match = len(good)

                src_pts = np.float32([pts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([pts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                # print("lon_pre, lat_pre:", lon_pre, lat_pre)

                lat_gt, lon_gt = df['lat'][i], df['lon'][i]

                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)


                print(RMSE)
        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)

def localization_village_NonOpt(args, test_ds, model, prediction):
    device = args.device

    MIN_MATCH_COUNT = 10

    # # dataset village
    Map_w = 4800
    Map_h = 2861
    lat1, lon1 = 47.05919722, 8.39368611  # 'bottom_left' E:454022.539, N:5211931.216
    lat2, lon2 = 47.07056667, 8.42165833  # 'top_right'   E:455978.326, N:5213187.329


    FLANN_INDEX_KDTREE = 1

    dataset_folder = test_ds.dataset_folder
    frame_path = dataset_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    error = []

    for i in range(test_ds.queries_num):
        img1_path = frame_path + df['file_name'][i]
        img1 = cv2.imread(img1_path, 0)  # query image
        img1 = cv2.resize(img1, (533, 400), interpolation=cv2.INTER_NEAREST) # village

        # img1 = cv2.resize(img1, (800, 600), interpolation=cv2.INTER_NEAREST) # gravel_pit
        image1 = img1
        img1 = img1.astype('float32') / 255.0
        outs1 = get_pts_and_des(model, img1 ,device=device)

        pts1, desc1 = outs1["pts"], outs1["desc"]
        print("###################################{}#############################".format(i))
        best_match = 0
        for j in range(10):

            index = prediction[i][j]
            img2_path = test_ds.database_paths[index]

            img2 = cv2.imread(img2_path, 0)
            image2 = img2# reference image
            img2 = img2.astype('float32') / 255.0
            outs2 = get_pts_and_des(model, img2, device=device)
            pts2, desc2 = outs2["pts"], outs2["desc"]



            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1.T, desc2.T, k=2)  # Finds the k best matches for each descriptor from a query set.

            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.85 * n.distance:
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT:

                best_match = len(good)
                # print(kp1[805].pt) # pixel coordinate
                # find the pixel coordinate of the matched query and template points
                src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                # src_pts = np.float32([pts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                # dst_pts = np.float32([pts1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)


                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                # print("lon_pre, lat_pre:", lon_pre, lat_pre)

                lat_gt, lon_gt = df['lat'][i], df['lon'][i]

                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)


                # cv2.drawMarker(image1, (int(x1), int(y1)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img1.jpg', image1)
                # cv2.drawMarker(image2, (int(trans_pt[0]), int(trans_pt[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img2.jpg', image2)
                #
                # # kp1 = pcv2.KeyPoint(x=pts1,y=2,size=3)
                # # kp1 = cv2.Keypoint(numpy.delete(pts1, 2, axis=0))
                # kpts1 = numpy.delete(pts1, 2, axis=0)
                # kpts1 =kpts1.T
                # cv_kpts1 = [cv2.KeyPoint(kpts1[i][0], kpts1[i][1], 1)
                #             for i in range(kpts1.shape[0])]
                #
                # kpts2 = numpy.delete(pts2, 2, axis=0)
                # kpts2 = kpts2.T
                # cv_kpts2 = [cv2.KeyPoint(kpts2[i][0], kpts2[i][1], 1)
                #             for i in range(kpts2.shape[0])]
                #
                #
                #
                # draw_params = dict(
                #     matchColor=(0, 255, 0),
                #     singlePointColor=None,
                #     matchesMask=matchesMask,
                #     flags=2)
                # img3 = cv2.drawMatches(image1, cv_kpts1, image2, cv_kpts2, good, None, **draw_params)
                # img1_name = (re.findall('\d+', img1_path))[0]
                # text = str(img1_name) + "  and  " + str(patch_h) + '_' + str(patch_w) + '   matches:' + str(len(matches)) \
                #        + 'good:' + str(len(good)) + '   RMSE:   ' + str(RMSE)
                # plt.text(0, -1, text)
                # plt.imshow(img3)
                # # file_name = str(img1_name) + '_' + str(patch_h) + '-' + str(patch_w) + '_' + str(j)
                # # plt.savefig('./result/rule/{}'.format(file_name))
                #
                # plt.show()

                print(RMSE)
        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)

def localization_village_matching(args, test_ds, model, prediction):
    device = args.device

    MIN_MATCH_COUNT = 10

    # # dataset village
    Map_w = 4800
    Map_h = 2861
    lat1, lon1 = 47.05919722, 8.39368611  # 'bottom_left' E:454022.539, N:5211931.216
    lat2, lon2 = 47.07056667, 8.42165833  # 'top_right'   E:455978.326, N:5213187.329


    FLANN_INDEX_KDTREE = 1

    dataset_folder = test_ds.dataset_folder
    frame_path = dataset_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    error = []

    for i in range(test_ds.queries_num):
        img1_path = frame_path + df['file_name'][i]
        img1 = cv2.imread(img1_path, 0)  # query image
        img1 = cv2.resize(img1, (533, 400), interpolation=cv2.INTER_NEAREST) # village

        image1 = img1
        img1 = img1.astype('float32') / 255.0
        outs1 = get_pts_and_des(model, img1 ,device=device)

        pts1, desc1 = outs1["pts"], outs1["desc"]
        print("###################################{}#############################".format(i))
        best_match = 0
        for img2_path in test_ds.database_paths:

            img2 = cv2.imread(img2_path, 0)
            image2 = img2# reference image
            img2 = img2.astype('float32') / 255.0
            outs2 = get_pts_and_des(model, img2, device=device)
            pts2, desc2 = outs2["pts"], outs2["desc"]



            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1.T, desc2.T, k=2) # village

            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.9 * n.distance:
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT:

                best_match = len(good)
                # print(kp1[805].pt) # pixel coordinate
                # find the pixel coordinate of the matched query and template points
                src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                # src_pts = np.float32([pts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                # dst_pts = np.float32([pts1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)


                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                # print("lon_pre, lat_pre:", lon_pre, lat_pre)

                lat_gt, lon_gt = df['lat'][i], df['lon'][i]

                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)

                print(RMSE)
        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)

def localization_village_SIFT_matching(args, test_ds, model, prediction):
    device = args.device

    MIN_MATCH_COUNT = 10

    # # dataset village
    Map_w = 4800
    Map_h = 2861
    lat1, lon1 = 47.05919722, 8.39368611  # 'bottom_left' E:454022.539, N:5211931.216
    lat2, lon2 = 47.07056667, 8.42165833  # 'top_right'   E:455978.326, N:5213187.329


    FLANN_INDEX_KDTREE = 1

    dataset_folder = test_ds.dataset_folder
    frame_path = dataset_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    error = []

    for i in range(test_ds.queries_num):
        img1_path = frame_path + df['file_name'][i]
        img1 = cv2.imread(img1_path, 0)  # query image
        img1 = cv2.resize(img1, (533, 400), interpolation=cv2.INTER_NEAREST) # village

        sift = cv2.SIFT_create()
        pts1, desc1 = sift.detectAndCompute(img1, None)


        print("###################################{}#############################".format(i))
        best_match = 0

        # for img2_path in test_ds.database_paths:
        for i in range(286):

            img2_path = test_ds.database_paths[i]

            img2 = cv2.imread(img2_path, 0)

            pts2, desc2 = sift.detectAndCompute(img2, None)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)  # Finds the k best matches for each descriptor from a query set.


            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.8 * n.distance:
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            # if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            if len(good) > MIN_MATCH_COUNT:

                best_match = len(good)

                src_pts = np.float32([pts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([pts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                # print("lon_pre, lat_pre:", lon_pre, lat_pre)

                lat_gt, lon_gt = df['lat'][i], df['lon'][i]

                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)


                print(RMSE)
        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)


def localization_gravelpit(args, test_ds, model, prediction):
    device = args.device
    initPosition = {'lon': 7.908635, 'lat': 47.11198889}

    def get_pts_and_des(model, img, device):
        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            outs = model(img.unsqueeze(0), 'SuperPoint')
            semi, coarse_desc = outs['semi'], outs['desc']

            heatmap = flattenDetection(semi, tensor=True)

            heatmap_np = heatmap.detach().cpu().numpy()

            pts_nms_batch = [getPtsFromHeatmap(h, conf_thresh=0.015, nms_dist=4) for h in heatmap_np]  #0.015 4
            ## subpixel
            pts = soft_argmax_points(heatmap_np, pts_nms_batch, patch_size= 5)

            desc_sparse = [sample_desc_from_points(outs['desc'], pts, device) for pts in pts_nms_batch]

            outs = {"pts": pts[0], "desc": desc_sparse[0]}
            return outs

    MIN_MATCH_COUNT = 10

    # dataset gravel_pit
    Map_w = 3355
    Map_h = 1852
    lat1, lon1 = 47.109375, 7.90523333  # 'bottom_left' 7.90523333, 47.109375
    lat2, lon2 = 47.11486111, 7.91988056  # 'top_right'   7.91988056, 47.11486111

    initPosition = {'lon': 7.916613333, 'lat': 47.11192944}
    mapCorner = {'left_lon': 7.90523333, 'right_lon': 7.91988056, 'bottom_lat': 47.109375, 'top_lat': 47.11486111}
    mapsize = {'height': 1852, 'width': 3355}
    patchsize = {'height': 600, 'width': 600}
    patchstep = 250

    dataset_folder = test_ds.dataset_folder
    frame_path = dataset_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    error = []

    current_patch = initPatch(initPosition, mapCorner, mapsize,  dataset_folder, patchstep)

    for i in range(1, test_ds.queries_num):

        img1_path = frame_path + df['file_name'][i]
        img1 = cv2.imread(img1_path, 0)  # query image

        img1 = cv2.resize(img1, (800, 600), interpolation=cv2.INTER_NEAREST) # gravel_pit
        image1 = img1
        img1 = img1.astype('float32') / 255.0
        outs1 = get_pts_and_des(model, img1 ,device=device)

        pts1, desc1 = outs1["pts"], outs1["desc"]
        print("###################################{}#############################".format(i))

        best_match = 0
        neighboring_patch = searchSpace(current_patch, patchstep, khop=1, folder=dataset_folder)
        neighboring_patch = list(set(neighboring_patch).intersection(test_ds.database_paths))

        # Ensure all patches exist
        # retrieval_patch = list(test_ds.database_paths[index] for index in prediction[i])
        # neighboring_patch = list(set(neighboring_patch).intersection(retrieval_patch))



        for neighbor in neighboring_patch:

            # index = prediction[i][j]
            img2_path = neighbor
            img2 = cv2.imread(img2_path, 0)
            image2 = img2
            img2 = img2.astype('float32') / 255.0
            outs2 = get_pts_and_des(model, img2, device=device)
            pts2, desc2 = outs2["pts"], outs2["desc"]

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1.T, desc2.T, k=2)
            # matches = bf.knnMatch(desc2.T, desc1.T, k=2)
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.9 * n.distance:
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT :

                best_match = len(good)
                src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                # src_pts = np.float32([(pts1.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                # dst_pts = np.float32([(pts2.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)


                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1

                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                lat_gt, lon_gt = df['lat'][i], df['lon'][i]

                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)


                # cv2.drawMarker(image1, (int(x1), int(y1)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img1.jpg', image1)
                # cv2.drawMarker(image2, (int(trans_pt[0]), int(trans_pt[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img2.jpg', image2)
                #
                # # kp1 = pcv2.KeyPoint(x=pts1,y=2,size=3)
                # # kp1 = cv2.Keypoint(numpy.delete(pts1, 2, axis=0))
                # kpts1 = numpy.delete(pts1, 2, axis=0)
                # kpts1 =kpts1.T
                # cv_kpts1 = [cv2.KeyPoint(kpts1[i][0], kpts1[i][1], 1)
                #             for i in range(kpts1.shape[0])]
                #
                # kpts2 = numpy.delete(pts2, 2, axis=0)
                # kpts2 = kpts2.T
                # cv_kpts2 = [cv2.KeyPoint(kpts2[i][0], kpts2[i][1], 1)
                #             for i in range(kpts2.shape[0])]
                #
                # draw_params = dict(
                #     matchColor=(0, 255, 0),
                #     singlePointColor=None,
                #     matchesMask=matchesMask,
                #     flags=2)
                # img3 = cv2.drawMatches(image1, cv_kpts1, image2, cv_kpts2, good, None, **draw_params)
                # img1_name = (re.findall('\d+', img1_path))[0]
                # text = str(img1_name) + "  and  " + str(patch_h) + '_' + str(patch_w) + '   matches:' + str(len(matches)) \
                #        + 'good:' + str(len(good)) + '   RMSE:   ' + str(RMSE)
                # plt.text(0, -1, text)
                # plt.imshow(img3)
                # plt.show()
                #
                # map = cv2.imread('/home/lhl/data/data/dataset/gravel_pit/map_gravel_pit.jpg', -1)
                # cv2.drawMarker(map, (int(trans_pt[0]+patch_w), int(trans_pt[1]+patch_h)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=300, thickness=3, line_type=cv2.LINE_AA)
                # # cv2.imwrite('village_map.jpg', map)
                # plt.imshow(map)
                #
                # plt.show()

                # current_patch = img2_path

                print(RMSE)

        Position = {'lon': df['lon'][i], 'lat': df['lat'][i] }
        current_patch = initPatch(Position, mapCorner, mapsize,  dataset_folder, patchstep)
        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)


def localization_gravelpit_match(args, test_ds, model, prediction):
    device = args.device

    MIN_MATCH_COUNT = 10

    # dataset gravel_pit
    Map_w = 3355
    Map_h = 1852
    lat1, lon1 = 47.109375, 7.90523333  # 'bottom_left' 7.90523333, 47.109375
    lat2, lon2 = 47.11486111, 7.91988056  # 'top_right'   7.91988056, 47.11486111

    initPosition = {'lon': 7.916613333, 'lat': 47.11192944}
    mapCorner = {'left_lon': 7.90523333, 'right_lon': 7.91988056, 'bottom_lat': 47.109375, 'top_lat': 47.11486111}
    mapsize = {'height': 1852, 'width': 3355}
    patchsize = {'height': 750, 'width': 1000}
    # patchstep = 250

    dataset_folder = test_ds.dataset_folder
    frame_path = dataset_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'
    map_path = '/home/lhl/data/data/dataset/gravel_pit/map_gravel_pit.jpg'
    df = pd.read_csv(gt_file)
    error = []

    # current_patch = initPatch(initPosition, mapCorner, mapsize, dataset_folder, patchstep)

    for i in range(0, test_ds.queries_num):

        img1_path = frame_path + df['file_name'][i]
        img1 = cv2.imread(img1_path, 0)  # query image
        img1 = cv2.resize(img1, (1600, 1200), interpolation=cv2.INTER_NEAREST)  # gravel_pit
        image1 = img1

        img1 = img1.astype('float32') / 255.0
        outs1 = get_pts_and_des(model, img1, device=device)

        pts1, desc1 = outs1["pts"], outs1["desc"]
        print("###################################{}#############################".format(i))

        best_match = 0
        Position = {'lon': df['lon'][i], 'lat': df['lat'][i]}
        img2, patch_h, patch_w = clipSpecificPatch(Position, mapCorner, mapsize, map_path, patchsize)
        image2 = img2


        img2 = img2.astype('float32') / 255.0
        outs2 = get_pts_and_des(model, img2, device=device)
        pts2, desc2 = outs2["pts"], outs2["desc"]

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1.T, desc2.T, k=2)

        good = []
        for m, n in matches:  # m and n are the two nearest matches for each querypoint.
            if m.distance < 0.9 * n.distance:
                # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                good.append(m)
        print('matches:', len(matches), 'good:', len(good))  # the number of good matched points

        if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
        # if len(good) > MIN_MATCH_COUNT :

            best_match = len(good)
            src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            (h1, w1) = img1.shape
            x1 = 0.5 * w1
            y1 = 0.5 * h1

            src_pt = np.insert([x1, y1], 2, values=1, axis=0)
            trans_pt = np.dot(H, src_pt)
            trans_pt = trans_pt / trans_pt[2]

            lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
            lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

            lat_gt, lon_gt = df['lat'][i], df['lon'][i]

            RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)

            print(RMSE)

            # cv2.drawMarker(image1, (int(x1), int(y1)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
            #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
            # cv2.imwrite('img1.jpg', image1)
            # cv2.drawMarker(image2, (int(trans_pt[0]), int(trans_pt[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS,
            #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
            # cv2.imwrite('img2.jpg', image2)
            #
            # # kp1 = pcv2.KeyPoint(x=pts1,y=2,size=3)
            # # kp1 = cv2.Keypoint(numpy.delete(pts1, 2, axis=0))
            # kpts1 = numpy.delete(pts1, 2, axis=0)
            # kpts1 =kpts1.T
            # cv_kpts1 = [cv2.KeyPoint(kpts1[i][0], kpts1[i][1], 1)
            #             for i in range(kpts1.shape[0])]
            #
            # kpts2 = numpy.delete(pts2, 2, axis=0)
            # kpts2 = kpts2.T
            # cv_kpts2 = [cv2.KeyPoint(kpts2[i][0], kpts2[i][1], 1)
            #             for i in range(kpts2.shape[0])]
            #
            # draw_params = dict(
            #     matchColor=(0, 255, 0),
            #     singlePointColor=None,
            #     matchesMask=matchesMask,
            #     flags=2)
            # img3 = cv2.drawMatches(image1, cv_kpts1, image2, cv_kpts2, good, None, **draw_params)
            # img1_name = (re.findall('\d+', img1_path))[0]
            # text = str(img1_name) + "  and  " + str(patch_h) + '_' + str(patch_w) + '   matches:' + str(len(matches)) \
            #        + 'good:' + str(len(good)) + '   RMSE:   ' + str(RMSE)
            # plt.text(0, -1, text)
            # plt.imshow(img3)
            # plt.show()
            #
            # map = cv2.imread('/home/lhl/data/data/dataset/gravel_pit/map_gravel_pit.jpg', -1)
            # cv2.drawMarker(map, (int(trans_pt[0]+patch_w), int(trans_pt[1]+patch_h)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
            #                markerSize=300, thickness=3, line_type=cv2.LINE_AA)
            # # cv2.imwrite('village_map.jpg', map)
            # plt.imshow(map)
            #
            # plt.show()


        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)


### OURS: Retrieval + SuperPoint + Jointly Optimized
def localization_Manchester(args, test_ds, model, prediction):
    device = args.device
    MIN_MATCH_COUNT = 10

    # dataset Manchester Aquatics Center
    # Map_w = 3370
    # Map_h = 2006
    # lat1, lon1 = 53.466619, -2.242382  # 'bottom_left'
    # lat2, lon2 = 53.470917, -2.230343  # 'top_right'
    # initP = {'lon': -2.24032, 'lat': 53.469095}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester ASDA
    # Map_w = 1278
    # Map_h = 3458
    # initP = {'lon': -2.2452716666666666, 'lat': 53.46532833333333}
    # lat1, lon1 = 53.459226, -2.249319  # 'bottom_left'
    # lat2, lon2 = 53.466568, -2.244801  # 'top_right'
    # patchstep = 300
    # patchsize = {'height': 400, 'width': 600}

    # dataset Manchester Business School
    # Map_w = 3324
    # Map_h = 1948
    # lat1, lon1 = 53.465320, -2.241710  # 'bottom_left'
    # lat2, lon2 = 53.468714, -2.231738  # 'top_right'
    # initP = {'lon': -2.2377783333333334, 'lat': 53.46703333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Energy Center
    # Map_w = 1744
    # Map_h = 1948
    # lat1, lon1 = 53.463926, -2.249017  # 'bottom_left'
    # lat2, lon2 = 53.466561, -2.245095  # 'top_right'
    # initP = {'lon': -2.248483333333333, 'lat': 53.46582166666666}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Holy Name Church
    # Map_w = 2314
    # Map_h = 1948
    # lat1, lon1 = 53.462834, -2.232979  # 'bottom_left'
    # lat2, lon2 = 53.466951, -2.224975  # 'top_right'
    # initP = {'lon': -2.232085, 'lat': 53.46494}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Hulme Park
    # Map_w = 2460
    # Map_h = 1948
    # lat1, lon1 = 53.467123, -2.252862  # 'bottom_left'
    # lat2, lon2 = 53.470593, -2.245797  # 'top_right'
    # initP = {'lon': -2.2505433333333333, 'lat': 53.46940333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Metropolitan University
    Map_w = 2254
    Map_h = 1948
    lat1, lon1 = 53.465042, -2.250405  # 'bottom_left'
    lat2, lon2 = 53.468178, -2.244545  # 'top_right'
    initP = {'lon': -2.2492916666666667, 'lat': 53.46724}
    patchstep = 400
    patchsize = {'height': 600, 'width': 800}

    # dataset Manchester Museum
    # Map_w = 2254
    # Map_h = 1948
    # lat1, lon1 = 53.464641, -2.235959  # 'bottom_left'
    # lat2, lon2 = 53.468706, -2.228272  # 'top_right'
    # initP = {'lon': -2.2335066666666665, 'lat': 53.466348333333336}
    # patchstep = 400
    # patchsize = {'height': 600, 'width': 800}

    dataset_folder = test_ds.dataset_folder
    frame_path = test_ds.frame_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    error = []

    for i in range(test_ds.queries_num):
        # img1_path = frame_path + df['file_name'][i]
        frame = test_ds.frames[i]
        df_row = df.loc[df['file_name'] == frame]

        img1_path = test_ds.queries_paths[i]

        img1 = cv2.imread(img1_path, 0)  # query image

        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # ASDA
        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # Business School
        img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)    #Energy Center

        image1 = img1
        img1 = img1.astype('float32') / 255.0
        outs1 = get_pts_and_des(model, img1 ,device=device)

        pts1, desc1 = outs1["pts"], outs1["desc"]
        print("###################################{}#############################".format(i))
        best_match = 0
        for j in range(20):

            index = prediction[i][j]
            img2_path = test_ds.database_paths[index]

            img2 = cv2.imread(img2_path, 0)
            image2 = img2# reference image
            img2 = img2.astype('float32') / 255.0
            outs2 = get_pts_and_des(model, img2, device=device)
            pts2, desc2 = outs2["pts"], outs2["desc"]

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1.T, desc2.T, k=2) # village


            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.9 * n.distance:
                    # ASDA 0.9
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT:

                best_match = len(good)

                src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                # print("lon_pre, lat_pre:", lon_pre, lat_pre)

                lat_gt, lon_gt = df_row['lat'], df_row['lon']

                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)
                print(RMSE)
        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)

###  Retrieval + SuperPoint + Frame by Frame
def localization_Manchester_test(args, test_ds, model, prediction):
    device = args.device
    MIN_MATCH_COUNT = 10

    # dataset Manchester Aquatics Center
    Map_w = 3370
    Map_h = 2006
    lat1, lon1 = 53.466619, -2.242382  # 'bottom_left'
    lat2, lon2 = 53.470917, -2.230343  # 'top_right'
    initP = {'lon': -2.24032, 'lat': 53.469095}
    patchstep = 300
    patchsize = {'height': 600, 'width': 800}

    # dataset Manchester ASDA
    # Map_w = 1278
    # Map_h = 3458
    # initP = {'lon': -2.2452716666666666, 'lat': 53.46532833333333}
    # lat1, lon1 = 53.459226, -2.249319  # 'bottom_left'
    # lat2, lon2 = 53.466568, -2.244801  # 'top_right'
    # patchstep = 300
    # patchsize = {'height': 400, 'width': 600}

    # dataset Manchester Business School
    # Map_w = 3324
    # Map_h = 1948
    # lat1, lon1 = 53.465320, -2.241710  # 'bottom_left'
    # lat2, lon2 = 53.468714, -2.231738  # 'top_right'
    # initP = {'lon': -2.2377783333333334, 'lat': 53.46703333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Energy Center
    # Map_w = 1744
    # Map_h = 1948
    # lat1, lon1 = 53.463926, -2.249017  # 'bottom_left'
    # lat2, lon2 = 53.466561, -2.245095  # 'top_right'
    # initP = {'lon': -2.248483333333333, 'lat': 53.46582166666666}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Holy Name Church
    # Map_w = 2314
    # Map_h = 1948
    # lat1, lon1 = 53.462834, -2.232979  # 'bottom_left'
    # lat2, lon2 = 53.466951, -2.224975  # 'top_right'
    # initP = {'lon': -2.232085, 'lat': 53.46494}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Hulme Park
    # Map_w = 2460
    # Map_h = 1948
    # lat1, lon1 = 53.467123, -2.252862  # 'bottom_left'
    # lat2, lon2 = 53.470593, -2.245797  # 'top_right'
    # initP = {'lon': -2.2505433333333333, 'lat': 53.46940333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Metropolitan University
    # Map_w = 2254
    # Map_h = 1948
    # lat1, lon1 = 53.465042, -2.250405  # 'bottom_left'
    # lat2, lon2 = 53.468178, -2.244545  # 'top_right'
    # initP = {'lon': -2.2492916666666667, 'lat': 53.46724}
    # patchstep = 400
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester Museum
    # Map_w = 2254
    # Map_h = 1948
    # lat1, lon1 = 53.464641, -2.235959  # 'bottom_left'
    # lat2, lon2 = 53.468706, -2.228272  # 'top_right'
    # initP = {'lon': -2.2335066666666665, 'lat': 53.466348333333336}
    # patchstep = 400
    # patchsize = {'height': 600, 'width': 800}

    dataset_folder = test_ds.dataset_folder
    frame_path = test_ds.frame_folder
    patches_path = test_ds.patch_folder
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    error = []
    mapCorner = {'left_lon': lon1, 'right_lon': lon2, 'bottom_lat': lat1, 'top_lat': lat2}
    mapsize = {'height': Map_h, 'width': Map_w}

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    current_patch = initPatch(initP, mapCorner, mapsize, dataset_folder, patchsize, patchstep)

    for i in range(1, test_ds.queries_num):
        # img1_path = frame_path + df['file_name'][i]
        frame = test_ds.frames[i]
        df_row = df.loc[df['file_name'] == frame].values.flatten().tolist()

        img1_path = test_ds.queries_paths[i]

        img1 = cv2.imread(img1_path, 0)  # query image

        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # ASDA
        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # Business School
        img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)    #Energy Center

        image1 = img1
        img1 = img1.astype('float32') / 255.0
        outs1 = get_pts_and_des(model, img1 ,device=device)

        pts1, desc1 = outs1["pts"], outs1["desc"]

        neighboring_patch = searchSpace(current_patch, patchstep, khop=1, folder=patches_path)
        neighboring_patch = list(set(neighboring_patch).intersection(test_ds.database_paths))

        print("###################################{}#############################".format(i))
        best_match = 0
        # for j in range(10):
        for neighbor in neighboring_patch:

            # index = prediction[i][j]
            img2_path = neighbor
            # index = prediction[i][j]
            # img2_path = test_ds.database_paths[index]

            img2 = cv2.imread(img2_path, 0)
            image2 = img2# reference image
            img2 = img2.astype('float32') / 255.0
            outs2 = get_pts_and_des(model, img2, device=device)
            pts2, desc2 = outs2["pts"], outs2["desc"]

            # bf = cv2.BFMatcher()
            # matches = bf.knnMatch(desc1.T, desc2.T, k=2) # village

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1.T, desc2.T, k=2)  # Finds the k best matches for each descriptor from a query set.



            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.9 * n.distance:
                    # ASDA 0.9
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT:
                best_match = len(good)
                src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                lat_gt, lon_gt = df_row[1], df_row[2]
                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)

                print(RMSE)

                # distance = GetDistance(lon_pre, lat_pre, initP['lon'], initP['lat'])
                # if distance < 300 and len(good) > best_match:
                #     best_match = len(good)
                #     lat_gt, lon_gt = df_row['lat'], df_row['lon']
                #     RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)
                #     print(RMSE)
                #
                #
                # cv2.drawMarker(image1, (int(x1), int(y1)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img1.jpg', image1)
                # cv2.drawMarker(image2, (int(trans_pt[0]), int(trans_pt[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img2.jpg', image2)
                #
                # # kp1 = pcv2.KeyPoint(x=pts1,y=2,size=3)
                # # kp1 = cv2.Keypoint(numpy.delete(pts1, 2, axis=0))
                # kpts1 = numpy.delete(pts1, 2, axis=0)
                # kpts1 =kpts1.T
                # cv_kpts1 = [cv2.KeyPoint(kpts1[i][0], kpts1[i][1], 1)
                #             for i in range(kpts1.shape[0])]
                #
                # kpts2 = numpy.delete(pts2, 2, axis=0)
                # kpts2 = kpts2.T
                # cv_kpts2 = [cv2.KeyPoint(kpts2[i][0], kpts2[i][1], 1)
                #             for i in range(kpts2.shape[0])]
                #
                #
                #
                # draw_params = dict(
                #     matchColor=(0, 255, 0),
                #     singlePointColor=None,
                #     matchesMask=matchesMask,
                #     flags=2)
                # img3 = cv2.drawMatches(image1, cv_kpts1, image2, cv_kpts2, good, None, **draw_params)
                # img1_name = (re.findall('\d+', img1_path))[0]
                # text = str(img1_name) + "  and  " + str(patch_h) + '_' + str(patch_w) + '   matches:' + str(len(matches)) \
                #        + 'good:' + str(len(good)) + '   RMSE:   ' + str(RMSE)
                # plt.text(0, -1, text)
                # plt.imshow(img3)
                # # file_name = str(img1_name) + '_' + str(patch_h) + '-' + str(patch_w) + '_' + str(j)
                # # plt.savefig('./result/rule/{}'.format(file_name))
                #
                # plt.show()
                #
                # print(RMSE)


        error.append(RMSE)

        Position = {'lon': lon_gt, 'lat': lat_gt}
        current_patch = initPatch(Position, mapCorner, mapsize, dataset_folder, patchsize, patchstep)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)

### Retrieval+SIFT
def localization_Manchester_SIFT(args, test_ds, model, prediction):
    device = args.device
    MIN_MATCH_COUNT = 10

    # dataset Manchester Aquatics Center
    # Map_w = 3370
    # Map_h = 2006
    # lat1, lon1 = 53.466619, -2.242382  # 'bottom_left'
    # lat2, lon2 = 53.470917, -2.230343  # 'top_right'
    # initP = {'lon': -2.24032, 'lat': 53.469095}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester ASDA
    # Map_w = 1278
    # Map_h = 3458
    # initP = {'lon': -2.2452716666666666, 'lat': 53.46532833333333}
    # lat1, lon1 = 53.459226, -2.249319  # 'bottom_left'
    # lat2, lon2 = 53.466568, -2.244801  # 'top_right'
    # patchstep = 300
    # patchsize = {'height': 400, 'width': 600}

    # dataset Manchester Business School
    Map_w = 3324
    Map_h = 1948
    lat1, lon1 = 53.465320, -2.241710  # 'bottom_left'
    lat2, lon2 = 53.468714, -2.231738  # 'top_right'
    initP = {'lon': -2.2377783333333334, 'lat': 53.46703333333333}
    patchstep = 200
    patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Energy Center
    # Map_w = 1744
    # Map_h = 1948
    # lat1, lon1 = 53.463926, -2.249017  # 'bottom_left'
    # lat2, lon2 = 53.466561, -2.245095  # 'top_right'
    # initP = {'lon': -2.248483333333333, 'lat': 53.46582166666666}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Holy Name Church
    # Map_w = 2314
    # Map_h = 1948
    # lat1, lon1 = 53.462834, -2.232979  # 'bottom_left'
    # lat2, lon2 = 53.466951, -2.224975  # 'top_right'
    # initP = {'lon': -2.232085, 'lat': 53.46494}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Hulme Park
    # Map_w = 2460
    # Map_h = 1948
    # lat1, lon1 = 53.467123, -2.252862  # 'bottom_left'
    # lat2, lon2 = 53.470593, -2.245797  # 'top_right'
    # initP = {'lon': -2.2505433333333333, 'lat': 53.46940333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Metropolitan University
    # Map_w = 2254
    # Map_h = 1948
    # lat1, lon1 = 53.465042, -2.250405  # 'bottom_left'
    # lat2, lon2 = 53.468178, -2.244545  # 'top_right'
    # initP = {'lon': -2.2492916666666667, 'lat': 53.46724}
    # patchstep = 400
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester Museum
    # Map_w = 2254
    # Map_h = 1948
    # lat1, lon1 = 53.464641, -2.235959  # 'bottom_left'
    # lat2, lon2 = 53.468706, -2.228272  # 'top_right'
    # initP = {'lon': -2.2335066666666665, 'lat': 53.466348333333336}
    # patchstep = 400
    # patchsize = {'height': 600, 'width': 800}

    dataset_folder = test_ds.dataset_folder
    frame_path = test_ds.frame_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    error = []

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)


    for i in range(test_ds.queries_num):
        # img1_path = frame_path + df['file_name'][i]
        frame = test_ds.frames[i]
        df_row = df.loc[df['file_name'] == frame]

        img1_path = test_ds.queries_paths[i]

        img1 = cv2.imread(img1_path, 0)  # query image

        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # ASDA
        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # Business School
        img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)    #Energy Center

        # Initial SIFT detector
        sift = cv2.SIFT_create()
        pts1, desc1 = sift.detectAndCompute(img1, None)

        # image1 = img1
        # img1 = img1.astype('float32') / 255.0
        # outs1 = get_pts_and_des(model, img1 ,device=device)
        # pts1, desc1 = outs1["pts"], outs1["desc"]

        print("###################################{}#############################".format(i))
        best_match = 0
        for j in range(10):

            index = prediction[i][j]
            img2_path = test_ds.database_paths[index]

            img2 = cv2.imread(img2_path, 0)
            pts2, desc2 = sift.detectAndCompute(img2, None)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)  # Finds the k best matches for each descriptor from a query set.

            # bf = cv2.BFMatcher()
            # matches = bf.knnMatch(desc1.T, desc2.T, k=2) # village


            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.8 * n.distance:
                    # ASDA 0.9
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT:

                best_match = len(good)

                # src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                # dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)

                src_pts = np.float32([pts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([pts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                # print("lon_pre, lat_pre:", lon_pre, lat_pre)

                lat_gt, lon_gt = df_row['lat'], df_row['lon']

                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)


                # cv2.drawMarker(image1, (int(x1), int(y1)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img1.jpg', image1)
                # cv2.drawMarker(image2, (int(trans_pt[0]), int(trans_pt[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img2.jpg', image2)
                #
                # # kp1 = pcv2.KeyPoint(x=pts1,y=2,size=3)
                # # kp1 = cv2.Keypoint(numpy.delete(pts1, 2, axis=0))
                # kpts1 = numpy.delete(pts1, 2, axis=0)
                # kpts1 =kpts1.T
                # cv_kpts1 = [cv2.KeyPoint(kpts1[i][0], kpts1[i][1], 1)
                #             for i in range(kpts1.shape[0])]
                #
                # kpts2 = numpy.delete(pts2, 2, axis=0)
                # kpts2 = kpts2.T
                # cv_kpts2 = [cv2.KeyPoint(kpts2[i][0], kpts2[i][1], 1)
                #             for i in range(kpts2.shape[0])]
                #
                #
                #
                # draw_params = dict(
                #     matchColor=(0, 255, 0),
                #     singlePointColor=None,
                #     matchesMask=matchesMask,
                #     flags=2)
                # img3 = cv2.drawMatches(image1, cv_kpts1, image2, cv_kpts2, good, None, **draw_params)
                # img1_name = (re.findall('\d+', img1_path))[0]
                # text = str(img1_name) + "  and  " + str(patch_h) + '_' + str(patch_w) + '   matches:' + str(len(matches)) \
                #        + 'good:' + str(len(good)) + '   RMSE:   ' + str(RMSE)
                # plt.text(0, -1, text)
                # plt.imshow(img3)
                # # file_name = str(img1_name) + '_' + str(patch_h) + '-' + str(patch_w) + '_' + str(j)
                # # plt.savefig('./result/rule/{}'.format(file_name))
                #
                # plt.show()
                #
                print(RMSE)
        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)

### Retrieval + SIFT +  Frame by Frame
def localization_Manchester_SIFT_test(args, test_ds, model, prediction):
    device = args.device
    MIN_MATCH_COUNT = 10

    # dataset Manchester Aquatics Center
    # Map_w = 3370
    # Map_h = 2006
    # lat1, lon1 = 53.466619, -2.242382  # 'bottom_left'
    # lat2, lon2 = 53.470917, -2.230343  # 'top_right'
    # initP = {'lon': -2.24032, 'lat': 53.469095}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester ASDA
    # Map_w = 1278
    # Map_h = 3458
    # initP = {'lon': -2.2452716666666666, 'lat': 53.46532833333333}
    # lat1, lon1 = 53.459226, -2.249319  # 'bottom_left'
    # lat2, lon2 = 53.466568, -2.244801  # 'top_right'
    # patchstep = 300
    # patchsize = {'height': 400, 'width': 600}

    # dataset Manchester Business School
    # Map_w = 3324
    # Map_h = 1948
    # lat1, lon1 = 53.465320, -2.241710  # 'bottom_left'
    # lat2, lon2 = 53.468714, -2.231738  # 'top_right'
    # initP = {'lon': -2.2377783333333334, 'lat': 53.46703333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # # dataset Manchester Energy Center
    # Map_w = 1744
    # Map_h = 1948
    # lat1, lon1 = 53.463926, -2.249017  # 'bottom_left'
    # lat2, lon2 = 53.466561, -2.245095  # 'top_right'
    # initP = {'lon': -2.248483333333333, 'lat': 53.46582166666666}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Holy Name Church
    # Map_w = 2314
    # Map_h = 1948
    # lat1, lon1 = 53.462834, -2.232979  # 'bottom_left'
    # lat2, lon2 = 53.466951, -2.224975  # 'top_right'
    # initP = {'lon': -2.232085, 'lat': 53.46494}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Hulme Park
    # Map_w = 2460
    # Map_h = 1948
    # lat1, lon1 = 53.467123, -2.252862  # 'bottom_left'
    # lat2, lon2 = 53.470593, -2.245797  # 'top_right'
    # initP = {'lon': -2.2505433333333333, 'lat': 53.46940333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Metropolitan University
    # Map_w = 2254
    # Map_h = 1948
    # lat1, lon1 = 53.465042, -2.250405  # 'bottom_left'
    # lat2, lon2 = 53.468178, -2.244545  # 'top_right'
    # initP = {'lon': -2.2492916666666667, 'lat': 53.46724}
    # patchstep = 400
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester Museum
    Map_w = 2254
    Map_h = 1948
    lat1, lon1 = 53.464641, -2.235959  # 'bottom_left'
    lat2, lon2 = 53.468706, -2.228272  # 'top_right'
    initP = {'lon': -2.2335066666666665, 'lat': 53.466348333333336}
    patchstep = 400
    patchsize = {'height': 600, 'width': 800}

    dataset_folder = test_ds.dataset_folder
    frame_path = test_ds.frame_folder
    patches_path = test_ds.patch_folder
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    error = []
    mapCorner = {'left_lon': lon1, 'right_lon': lon2, 'bottom_lat': lat1, 'top_lat': lat2}
    mapsize = {'height': Map_h, 'width': Map_w}

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    current_patch = initPatch(initP, mapCorner, mapsize, dataset_folder, patchsize, patchstep)

    for i in range(1, test_ds.queries_num):
        # img1_path = frame_path + df['file_name'][i]
        frame = test_ds.frames[i]
        df_row = df.loc[df['file_name'] == frame].values.flatten().tolist()

        img1_path = test_ds.queries_paths[i]

        img1 = cv2.imread(img1_path, 0)  # query image

        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # ASDA
        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # Business School
        img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # Energy Center

        # Initial SIFT detector
        sift = cv2.SIFT_create()
        pts1, desc1 = sift.detectAndCompute(img1, None)



        neighboring_patch = searchSpace(current_patch, patchstep, khop=1, folder=patches_path)
        neighboring_patch = list(set(neighboring_patch).intersection(test_ds.database_paths))

        print("###################################{}#############################".format(i))
        best_match = 0
        # for j in range(10):
        for neighbor in neighboring_patch:

            # index = prediction[i][j]
            img2_path = neighbor
            # index = prediction[i][j]
            # img2_path = test_ds.database_paths[index]

            img2 = cv2.imread(img2_path, 0)

            pts2, desc2 = sift.detectAndCompute(img2, None)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)  # Finds the k best matches for each descriptor from a query set.



            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.9 * n.distance:
                    # ASDA 0.9
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT:
                best_match = len(good)
                src_pts = np.float32([pts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([pts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                lat_gt, lon_gt = df_row[1], df_row[2]
                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)

                print(RMSE)

                # distance = GetDistance(lon_pre, lat_pre, initP['lon'], initP['lat'])
                # if distance < 300 and len(good) > best_match:
                #     best_match = len(good)
                #     lat_gt, lon_gt = df_row['lat'], df_row['lon']
                #     RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)
                #     print(RMSE)
                #
                #
                # cv2.drawMarker(image1, (int(x1), int(y1)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img1.jpg', image1)
                # cv2.drawMarker(image2, (int(trans_pt[0]), int(trans_pt[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img2.jpg', image2)
                #
                # # kp1 = pcv2.KeyPoint(x=pts1,y=2,size=3)
                # # kp1 = cv2.Keypoint(numpy.delete(pts1, 2, axis=0))
                # kpts1 = numpy.delete(pts1, 2, axis=0)
                # kpts1 =kpts1.T
                # cv_kpts1 = [cv2.KeyPoint(kpts1[i][0], kpts1[i][1], 1)
                #             for i in range(kpts1.shape[0])]
                #
                # kpts2 = numpy.delete(pts2, 2, axis=0)
                # kpts2 = kpts2.T
                # cv_kpts2 = [cv2.KeyPoint(kpts2[i][0], kpts2[i][1], 1)
                #             for i in range(kpts2.shape[0])]
                #
                #
                #
                # draw_params = dict(
                #     matchColor=(0, 255, 0),
                #     singlePointColor=None,
                #     matchesMask=matchesMask,
                #     flags=2)
                # img3 = cv2.drawMatches(image1, cv_kpts1, image2, cv_kpts2, good, None, **draw_params)
                # img1_name = (re.findall('\d+', img1_path))[0]
                # text = str(img1_name) + "  and  " + str(patch_h) + '_' + str(patch_w) + '   matches:' + str(len(matches)) \
                #        + 'good:' + str(len(good)) + '   RMSE:   ' + str(RMSE)
                # plt.text(0, -1, text)
                # plt.imshow(img3)
                # # file_name = str(img1_name) + '_' + str(patch_h) + '-' + str(patch_w) + '_' + str(j)
                # # plt.savefig('./result/rule/{}'.format(file_name))
                #
                # plt.show()
                #
                # print(RMSE)


        error.append(RMSE)

        Position = {'lon': lon_gt, 'lat': lat_gt}
        current_patch = initPatch(Position, mapCorner, mapsize, dataset_folder, patchsize, patchstep)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)

### Only Matching
def localization_Manchester_matching(args, test_ds, model, prediction):
    device = args.device
    MIN_MATCH_COUNT = 10

    # dataset Manchester Aquatics Center
    # Map_w = 3370
    # Map_h = 2006
    # lat1, lon1 = 53.466619, -2.242382  # 'bottom_left'
    # lat2, lon2 = 53.470917, -2.230343  # 'top_right'
    # initP = {'lon': -2.24032, 'lat': 53.469095}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester ASDA
    # Map_w = 1278
    # Map_h = 3458
    # initP = {'lon': -2.2452716666666666, 'lat': 53.46532833333333}
    # lat1, lon1 = 53.459226, -2.249319  # 'bottom_left'
    # lat2, lon2 = 53.466568, -2.244801  # 'top_right'
    # patchstep = 300
    # patchsize = {'height': 400, 'width': 600}

    # dataset Manchester Business School
    # Map_w = 3324
    # Map_h = 1948
    # lat1, lon1 = 53.465320, -2.241710  # 'bottom_left'
    # lat2, lon2 = 53.468714, -2.231738  # 'top_right'
    # initP = {'lon': -2.2377783333333334, 'lat': 53.46703333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Energy Center
    # Map_w = 1744
    # Map_h = 1948
    # lat1, lon1 = 53.463926, -2.249017  # 'bottom_left'
    # lat2, lon2 = 53.466561, -2.245095  # 'top_right'
    # initP = {'lon': -2.248483333333333, 'lat': 53.46582166666666}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Holy Name Church
    # Map_w = 2314
    # Map_h = 1948
    # lat1, lon1 = 53.462834, -2.232979  # 'bottom_left'
    # lat2, lon2 = 53.466951, -2.224975  # 'top_right'
    # initP = {'lon': -2.232085, 'lat': 53.46494}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Hulme Park
    # Map_w = 2460
    # Map_h = 1948
    # lat1, lon1 = 53.467123, -2.252862  # 'bottom_left'
    # lat2, lon2 = 53.470593, -2.245797  # 'top_right'
    # initP = {'lon': -2.2505433333333333, 'lat': 53.46940333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Metropolitan University
    # Map_w = 2254
    # Map_h = 1948
    # lat1, lon1 = 53.465042, -2.250405  # 'bottom_left'
    # lat2, lon2 = 53.468178, -2.244545  # 'top_right'
    # initP = {'lon': -2.2492916666666667, 'lat': 53.46724}
    # patchstep = 400
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester Museum
    Map_w = 2254
    Map_h = 1948
    lat1, lon1 = 53.464641, -2.235959  # 'bottom_left'
    lat2, lon2 = 53.468706, -2.228272  # 'top_right'
    initP = {'lon': -2.2335066666666665, 'lat': 53.466348333333336}
    patchstep = 400
    patchsize = {'height': 600, 'width': 800}

    dataset_folder = test_ds.dataset_folder
    frame_path = test_ds.frame_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    error = []

    for i in range(test_ds.queries_num):
        # img1_path = frame_path + df['file_name'][i]
        frame = test_ds.frames[i]
        df_row = df.loc[df['file_name'] == frame]

        img1_path = test_ds.queries_paths[i]

        img1 = cv2.imread(img1_path, 0)  # query image

        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # ASDA
        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # Business School
        img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)    #Energy Center

        image1 = img1
        img1 = img1.astype('float32') / 255.0
        outs1 = get_pts_and_des(model, img1 ,device=device)

        pts1, desc1 = outs1["pts"], outs1["desc"]
        print("###################################{}#############################".format(i))
        best_match = 0
        for img2_path in test_ds.database_paths:

            img2 = cv2.imread(img2_path, 0)
            image2 = img2# reference image
            img2 = img2.astype('float32') / 255.0
            outs2 = get_pts_and_des(model, img2, device=device)
            pts2, desc2 = outs2["pts"], outs2["desc"]

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1.T, desc2.T, k=2) # village


            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.9 * n.distance:
                    # ASDA 0.9
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT:

                best_match = len(good)

                src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                # print("lon_pre, lat_pre:", lon_pre, lat_pre)

                lat_gt, lon_gt = df_row['lat'], df_row['lon']

                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)
                print(RMSE)
        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)

### Retrieval + SuperPoint + init
def localization_Manchester_init(args, test_ds, model, prediction):
    device = args.device
    MIN_MATCH_COUNT = 10

    # dataset Manchester Aquatics Center
    # Map_w = 3370
    # Map_h = 2006
    # lat1, lon1 = 53.466619, -2.242382  # 'bottom_left'
    # lat2, lon2 = 53.470917, -2.230343  # 'top_right'
    # initP = {'lon': -2.24032, 'lat': 53.469095}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester ASDA
    # Map_w = 1278
    # Map_h = 3458
    # initP = {'lon': -2.2452716666666666, 'lat': 53.46532833333333}
    # lat1, lon1 = 53.459226, -2.249319  # 'bottom_left'
    # lat2, lon2 = 53.466568, -2.244801  # 'top_right'
    # patchstep = 300
    # patchsize = {'height': 400, 'width': 600}

    # dataset Manchester Business School
    # Map_w = 3324
    # Map_h = 1948
    # lat1, lon1 = 53.465320, -2.241710  # 'bottom_left'
    # lat2, lon2 = 53.468714, -2.231738  # 'top_right'
    # initP = {'lon': -2.2377783333333334, 'lat': 53.46703333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Energy Center
    # Map_w = 1744
    # Map_h = 1948
    # lat1, lon1 = 53.463926, -2.249017  # 'bottom_left'
    # lat2, lon2 = 53.466561, -2.245095  # 'top_right'
    # initP = {'lon': -2.248483333333333, 'lat': 53.46582166666666}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Holy Name Church
    Map_w = 2314
    Map_h = 1948
    lat1, lon1 = 53.462834, -2.232979  # 'bottom_left'
    lat2, lon2 = 53.466951, -2.224975  # 'top_right'
    initP = {'lon': -2.232085, 'lat': 53.46494}
    patchstep = 300
    patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Hulme Park
    # Map_w = 2460
    # Map_h = 1948
    # lat1, lon1 = 53.467123, -2.252862  # 'bottom_left'
    # lat2, lon2 = 53.470593, -2.245797  # 'top_right'
    # initP = {'lon': -2.2505433333333333, 'lat': 53.46940333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Metropolitan University
    # Map_w = 2254
    # Map_h = 1948
    # lat1, lon1 = 53.465042, -2.250405  # 'bottom_left'
    # lat2, lon2 = 53.468178, -2.244545  # 'top_right'
    # initP = {'lon': -2.2492916666666667, 'lat': 53.46724}
    # patchstep = 400
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester Museum
    # Map_w = 2254
    # Map_h = 1948
    # lat1, lon1 = 53.464641, -2.235959  # 'bottom_left'
    # lat2, lon2 = 53.468706, -2.228272  # 'top_right'
    # initP = {'lon': -2.2335066666666665, 'lat': 53.466348333333336}
    # patchstep = 400
    # patchsize = {'height': 600, 'width': 800}

    dataset_folder = test_ds.dataset_folder
    frame_path = test_ds.frame_folder
    patches_path = test_ds.patch_folder
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    error = []
    mapCorner = {'left_lon': lon1, 'right_lon': lon2, 'bottom_lat': lat1, 'top_lat': lat2}
    mapsize = {'height': Map_h, 'width': Map_w}

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    current_patch = initPatch(initP, mapCorner, mapsize, dataset_folder, patchsize, patchstep)
    neighboring_patch = searchSpace(current_patch, patchstep, khop=1, folder=patches_path)
    candidate_patch = list(set(neighboring_patch).intersection(test_ds.database_paths))
    previous_position = initP
    for i in range(1, test_ds.queries_num):
        # img1_path = frame_path + df['file_name'][i]
        frame = test_ds.frames[i]
        df_row = df.loc[df['file_name'] == frame].values.flatten().tolist()

        img1_path = test_ds.queries_paths[i]

        img1 = cv2.imread(img1_path, 0)  # query image

        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # ASDA
        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # Business School
        img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)    #Energy Center

        image1 = img1
        img1 = img1.astype('float32') / 255.0
        outs1 = get_pts_and_des(model, img1 ,device=device)

        pts1, desc1 = outs1["pts"], outs1["desc"]


        print("###################################{}#############################".format(i))
        best_match = 0
        # for j in range(10):
        for neighbor in candidate_patch:

            # index = prediction[i][j]
            img2_path = neighbor
            # index = prediction[i][j]
            # img2_path = test_ds.database_paths[index]

            img2 = cv2.imread(img2_path, 0)
            image2 = img2# reference image
            img2 = img2.astype('float32') / 255.0
            outs2 = get_pts_and_des(model, img2, device=device)
            pts2, desc2 = outs2["pts"], outs2["desc"]

            # bf = cv2.BFMatcher()
            # matches = bf.knnMatch(desc1.T, desc2.T, k=2) # village

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1.T, desc2.T, k=2)  # Finds the k best matches for each descriptor from a query set.



            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.9 * n.distance:
                    # ASDA 0.9
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT:
                best_match = len(good)
                src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                lat_gt, lon_gt = df_row[1], df_row[2]
                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)

                print(RMSE)

        error.append(RMSE)

        Predicted_Position = {'lon': lon_pre, 'lat': lat_pre}
        distance = GetDistance(lon_pre, lat_pre, previous_position['lon'], previous_position['lat'])
        ### Others150  Museum 300  M_U 80
        if distance < 100:
            current_patch = initPatch(Predicted_Position, mapCorner, mapsize, dataset_folder, patchsize, patchstep)
            previous_position = Predicted_Position
            neighboring_patch = searchSpace(current_patch, patchstep, khop=1, folder=patches_path)
            candidate_patch = list(set(neighboring_patch).intersection(test_ds.database_paths))
        else:
            current_patch = initPatch(Predicted_Position, mapCorner, mapsize, dataset_folder, patchsize, patchstep)
            current_neighboring_patch = searchSpace(current_patch, patchstep, khop=1, folder=patches_path)
            current_neighboring_patch = list(set(current_neighboring_patch).intersection(test_ds.database_paths))

            previous_patch = initPatch(previous_position, mapCorner, mapsize, dataset_folder, patchsize, patchstep)
            previous_neighboring_patch = searchSpace(previous_patch, patchstep, khop=1, folder=patches_path)
            previous_neighboring_patch = list(set(previous_neighboring_patch).intersection(test_ds.database_paths))

            candidate_patch = list(set(previous_neighboring_patch).union(set(current_neighboring_patch)))

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)

### Retrieval + SuperPoint + NonOptimized
def localization_Manchester_NonOpt(args, test_ds, model, prediction):
    device = args.device
    MIN_MATCH_COUNT = 10

    # dataset Manchester Aquatics Center
    # Map_w = 3370
    # Map_h = 2006
    # lat1, lon1 = 53.466619, -2.242382  # 'bottom_left'
    # lat2, lon2 = 53.470917, -2.230343  # 'top_right'
    # initP = {'lon': -2.24032, 'lat': 53.469095}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester ASDA
    # Map_w = 1278
    # Map_h = 3458
    # initP = {'lon': -2.2452716666666666, 'lat': 53.46532833333333}
    # lat1, lon1 = 53.459226, -2.249319  # 'bottom_left'
    # lat2, lon2 = 53.466568, -2.244801  # 'top_right'
    # patchstep = 300
    # patchsize = {'height': 400, 'width': 600}

    # dataset Manchester Business School
    # Map_w = 3324
    # Map_h = 1948
    # lat1, lon1 = 53.465320, -2.241710  # 'bottom_left'
    # lat2, lon2 = 53.468714, -2.231738  # 'top_right'
    # initP = {'lon': -2.2377783333333334, 'lat': 53.46703333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Energy Center
    # Map_w = 1744
    # Map_h = 1948
    # lat1, lon1 = 53.463926, -2.249017  # 'bottom_left'
    # lat2, lon2 = 53.466561, -2.245095  # 'top_right'
    # initP = {'lon': -2.248483333333333, 'lat': 53.46582166666666}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Holy Name Church
    # Map_w = 2314
    # Map_h = 1948
    # lat1, lon1 = 53.462834, -2.232979  # 'bottom_left'
    # lat2, lon2 = 53.466951, -2.224975  # 'top_right'
    # initP = {'lon': -2.232085, 'lat': 53.46494}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Hulme Park
    # Map_w = 2460
    # Map_h = 1948
    # lat1, lon1 = 53.467123, -2.252862  # 'bottom_left'
    # lat2, lon2 = 53.470593, -2.245797  # 'top_right'
    # initP = {'lon': -2.2505433333333333, 'lat': 53.46940333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Metropolitan University
    # Map_w = 2254
    # Map_h = 1948
    # lat1, lon1 = 53.465042, -2.250405  # 'bottom_left'
    # lat2, lon2 = 53.468178, -2.244545  # 'top_right'
    # initP = {'lon': -2.2492916666666667, 'lat': 53.46724}
    # patchstep = 400
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester Museum
    Map_w = 2254
    Map_h = 1948
    lat1, lon1 = 53.464641, -2.235959  # 'bottom_left'
    lat2, lon2 = 53.468706, -2.228272  # 'top_right'
    initP = {'lon': -2.2335066666666665, 'lat': 53.466348333333336}
    patchstep = 400
    patchsize = {'height': 600, 'width': 800}

    dataset_folder = test_ds.dataset_folder
    frame_path = test_ds.frame_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    error = []

    for i in range(test_ds.queries_num):
        # img1_path = frame_path + df['file_name'][i]
        frame = test_ds.frames[i]
        df_row = df.loc[df['file_name'] == frame]

        img1_path = test_ds.queries_paths[i]

        img1 = cv2.imread(img1_path, 0)  # query image

        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # ASDA
        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # Business School
        img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)    #Energy Center

        image1 = img1
        img1 = img1.astype('float32') / 255.0
        outs1 = get_pts_and_des(model, img1 ,device=device)

        pts1, desc1 = outs1["pts"], outs1["desc"]
        print("###################################{}#############################".format(i))
        best_match = 0
        for j in range(10):

            index = prediction[i][j]
            img2_path = test_ds.database_paths[index]

            img2 = cv2.imread(img2_path, 0)
            image2 = img2# reference image
            img2 = img2.astype('float32') / 255.0
            outs2 = get_pts_and_des(model, img2, device=device)
            pts2, desc2 = outs2["pts"], outs2["desc"]

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1.T, desc2.T, k=2) # village


            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.9 * n.distance:
                    # ASDA 0.9
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT:

                best_match = len(good)

                src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                # print("lon_pre, lat_pre:", lon_pre, lat_pre)

                lat_gt, lon_gt = df_row['lat'], df_row['lon']

                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)
                print(RMSE)
        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)

### Only Matchign + SIFT
### Retrieval+SIFT
def localization_Manchester_matching_SIFT(args, test_ds, model, prediction):
    device = args.device
    MIN_MATCH_COUNT = 10

    # dataset Manchester Aquatics Center
    # Map_w = 3370
    # Map_h = 2006
    # lat1, lon1 = 53.466619, -2.242382  # 'bottom_left'
    # lat2, lon2 = 53.470917, -2.230343  # 'top_right'
    # initP = {'lon': -2.24032, 'lat': 53.469095}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester ASDA
    # Map_w = 1278
    # Map_h = 3458
    # initP = {'lon': -2.2452716666666666, 'lat': 53.46532833333333}
    # lat1, lon1 = 53.459226, -2.249319  # 'bottom_left'
    # lat2, lon2 = 53.466568, -2.244801  # 'top_right'
    # patchstep = 300
    # patchsize = {'height': 400, 'width': 600}

    # dataset Manchester Business School
    # Map_w = 3324
    # Map_h = 1948
    # lat1, lon1 = 53.465320, -2.241710  # 'bottom_left'
    # lat2, lon2 = 53.468714, -2.231738  # 'top_right'
    # initP = {'lon': -2.2377783333333334, 'lat': 53.46703333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Energy Center
    # Map_w = 1744
    # Map_h = 1948
    # lat1, lon1 = 53.463926, -2.249017  # 'bottom_left'
    # lat2, lon2 = 53.466561, -2.245095  # 'top_right'
    # initP = {'lon': -2.248483333333333, 'lat': 53.46582166666666}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Holy Name Church
    # Map_w = 2314
    # Map_h = 1948
    # lat1, lon1 = 53.462834, -2.232979  # 'bottom_left'
    # lat2, lon2 = 53.466951, -2.224975  # 'top_right'
    # initP = {'lon': -2.232085, 'lat': 53.46494}
    # patchstep = 300
    # patchsize = {'height': 600, 'width': 800}

    # # dataset Manchester Hulme Park
    # Map_w = 2460
    # Map_h = 1948
    # lat1, lon1 = 53.467123, -2.252862  # 'bottom_left'
    # lat2, lon2 = 53.470593, -2.245797  # 'top_right'
    # initP = {'lon': -2.2505433333333333, 'lat': 53.46940333333333}
    # patchstep = 200
    # patchsize = {'height': 400, 'width': 600}

    # # dataset Manchester Metropolitan University
    # Map_w = 2254
    # Map_h = 1948
    # lat1, lon1 = 53.465042, -2.250405  # 'bottom_left'
    # lat2, lon2 = 53.468178, -2.244545  # 'top_right'
    # initP = {'lon': -2.2492916666666667, 'lat': 53.46724}
    # patchstep = 400
    # patchsize = {'height': 600, 'width': 800}

    # dataset Manchester Museum
    Map_w = 2254
    Map_h = 1948
    lat1, lon1 = 53.464641, -2.235959  # 'bottom_left'
    lat2, lon2 = 53.468706, -2.228272  # 'top_right'
    initP = {'lon': -2.2335066666666665, 'lat': 53.466348333333336}
    patchstep = 400
    patchsize = {'height': 600, 'width': 800}

    dataset_folder = test_ds.dataset_folder
    frame_path = test_ds.frame_folder + '/frames/'
    patches_path = dataset_folder + '/patch/'
    gt_file = dataset_folder + '/coordinates.csv'

    df = pd.read_csv(gt_file)
    error = []

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)


    for i in range(test_ds.queries_num):
        # img1_path = frame_path + df['file_name'][i]
        frame = test_ds.frames[i]
        df_row = df.loc[df['file_name'] == frame]

        img1_path = test_ds.queries_paths[i]

        img1 = cv2.imread(img1_path, 0)  # query image

        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # ASDA
        # img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)  # Business School
        img1 = cv2.resize(img1, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)    #Energy Center

        # Initial SIFT detector
        sift = cv2.SIFT_create()
        pts1, desc1 = sift.detectAndCompute(img1, None)

        # image1 = img1
        # img1 = img1.astype('float32') / 255.0
        # outs1 = get_pts_and_des(model, img1 ,device=device)
        # pts1, desc1 = outs1["pts"], outs1["desc"]

        print("###################################{}#############################".format(i))
        best_match = 0
        for img2_path in test_ds.database_paths:

            img2 = cv2.imread(img2_path, 0)


            pts2, desc2 = sift.detectAndCompute(img2, None)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)  # Finds the k best matches for each descriptor from a query set.

            # bf = cv2.BFMatcher()
            # matches = bf.knnMatch(desc1.T, desc2.T, k=2) # village


            # store all the good matches as per lows ratio test
            good = []
            for m, n in matches: # m and n are the two nearest matches for each querypoint.
                if m.distance < 0.8 * n.distance:
                    # ASDA 0.9
                    # if the first nearest matches is smaller than the second one by a large margin, it is a good match
                    good.append(m)
            print(img1_path, img2_path)
            print('matches:', len(matches), 'good:', len(good)) # the number of good matched points

            if len(good) > MIN_MATCH_COUNT and len(good) > best_match:
            # if len(good) > MIN_MATCH_COUNT:

                best_match = len(good)

                # src_pts = np.float32([(pts1.T)[m.queryIdx, 0:2] for m in good]).reshape(-1, 1, 2)
                # dst_pts = np.float32([(pts2.T)[m.trainIdx, 0:2] for m in good]).reshape(-1, 1, 2)

                src_pts = np.float32([pts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([pts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                (h1, w1) = img1.shape
                x1 = 0.5* w1
                y1 = 0.5 * h1
                # print('x1,y1:', x1,y1)
                src_pt = np.insert([x1, y1], 2, values=1, axis=0)
                trans_pt = np.dot(H, src_pt)
                trans_pt = trans_pt / trans_pt[2]
                # print(src_pt, trans_pt)

                patch_corner = re.findall('\d+', img2_path)
                patch_w = int(patch_corner[1])
                patch_h = int(patch_corner[0])

                lon_pre = (int(trans_pt[0]) + patch_w) * (lon2 - lon1) / Map_w + lon1
                lat_pre = (Map_h - (int(trans_pt[1]) + patch_h)) * (lat2 - lat1) / Map_h + lat1

                # print("lon_pre, lat_pre:", lon_pre, lat_pre)

                lat_gt, lon_gt = df_row['lat'], df_row['lon']

                RMSE = GetDistance(lon_pre, lat_pre, lon_gt, lat_gt)


                # cv2.drawMarker(image1, (int(x1), int(y1)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img1.jpg', image1)
                # cv2.drawMarker(image2, (int(trans_pt[0]), int(trans_pt[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                #                markerSize=200, thickness=5, line_type=cv2.LINE_AA)
                # cv2.imwrite('img2.jpg', image2)
                #
                # # kp1 = pcv2.KeyPoint(x=pts1,y=2,size=3)
                # # kp1 = cv2.Keypoint(numpy.delete(pts1, 2, axis=0))
                # kpts1 = numpy.delete(pts1, 2, axis=0)
                # kpts1 =kpts1.T
                # cv_kpts1 = [cv2.KeyPoint(kpts1[i][0], kpts1[i][1], 1)
                #             for i in range(kpts1.shape[0])]
                #
                # kpts2 = numpy.delete(pts2, 2, axis=0)
                # kpts2 = kpts2.T
                # cv_kpts2 = [cv2.KeyPoint(kpts2[i][0], kpts2[i][1], 1)
                #             for i in range(kpts2.shape[0])]
                #
                #
                #
                # draw_params = dict(
                #     matchColor=(0, 255, 0),
                #     singlePointColor=None,
                #     matchesMask=matchesMask,
                #     flags=2)
                # img3 = cv2.drawMatches(image1, cv_kpts1, image2, cv_kpts2, good, None, **draw_params)
                # img1_name = (re.findall('\d+', img1_path))[0]
                # text = str(img1_name) + "  and  " + str(patch_h) + '_' + str(patch_w) + '   matches:' + str(len(matches)) \
                #        + 'good:' + str(len(good)) + '   RMSE:   ' + str(RMSE)
                # plt.text(0, -1, text)
                # plt.imshow(img3)
                # # file_name = str(img1_name) + '_' + str(patch_h) + '-' + str(patch_w) + '_' + str(j)
                # # plt.savefig('./result/rule/{}'.format(file_name))
                #
                # plt.show()
                #
                print(RMSE)
        error.append(RMSE)

    print(error)
    print("############# Avg Error #############")
    error_avg = np.mean(error)
    print(error_avg)