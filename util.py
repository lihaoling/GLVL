from math import pi
from imgaug import augmenters as iaa
import cv2
import re
import torch
import shutil
import logging
import torchscan
import numpy as np
from collections import OrderedDict
from os.path import join
import collections
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os

import datasets_ws

def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_flops(model, input_shape=(480, 640)):
    """Return the FLOPs as a string, such as '22.33 GFLOPs'"""
    assert len(input_shape) == 2, f"input_shape should have len==2, but it's {input_shape}"
    module_info = torchscan.crawl_module(model, (3, input_shape[0], input_shape[1]))
    output = torchscan.utils.format_info(module_info)
    return re.findall("Floating Point Operations on forward: (.*)\n", output)[0]


def save_checkpoint(save_path, net_state, epoch, filename='checkpoint.pth'):
    file_prefix = ['superPointNet']
    # torch.save(net_state, save_path)
    if epoch % 2000 == 0:
        filename = '{}_{}_{}'.format(file_prefix[0], str(epoch), filename)
        path = Path(save_path)
        save_path = path / 'checkpoints'
        os.makedirs(save_path, exist_ok=True)
        torch.save(net_state, save_path/filename)
        print("save checkpoint to ", filename)
    pass

def save_checkpoint_train(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))


def retrieval_save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))



def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    return model


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current_best_R@5 = {best_r5:.1f}")
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r5, start_epoch_num, not_improved_num



def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    def to_3d(img):
        if len(img.shape) == 2:
            img = img[np.newaxis, ...]
        return img
    img_r, img_g, img_gray = to_3d(img_r), to_3d(img_g), to_3d(img_gray)
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def thd_img(img, thd=0.015):
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def save_path_formatter(args, parser):
    print("todo: save path")
    return Path('.')
    pass
'''
def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    # data_folder_name = str(Path(args_dict['data']).normpath().name)
    data_folder_name = str(Path(args_dict['data']))
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['sequence_length'] = 'seq'
    keys_with_prefix['rotation_mode'] = 'rot_'
    keys_with_prefix['padding_mode'] = 'padding_'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'
    keys_with_prefix['photo_loss_weight'] = 'p'
    keys_with_prefix['mask_loss_weight'] = 'm'
    keys_with_prefix['smooth_loss_weight'] = 's'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp

    # return ''
'''


def tensor2array(tensor, max_value=255, colormap='rainbow', channel_first=True):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if int(cv2.__version__[0]) >= 3:
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)
        if channel_first:
            array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
        if not channel_first:
            array = array.transpose(1, 2, 0)
    return array


# from utils.utils import find_files_with_ext
def find_files_with_ext(directory, extension='.npz'):
    # print(os.listdir(directory))
    list_of_files = []
    import os
    if extension == ".npz":
        for l in os.listdir(directory):
            if l.endswith(extension):
                list_of_files.append(l)
                # print(l)
        return list_of_files


def load_checkpoint(load_path, filename='checkpoint.pth.tar'):
    file_prefix = ['superPointNet']
    filename = '{}__{}'.format(file_prefix[0], filename)
    # torch.save(net_state, save_path)
    checkpoint = torch.load(load_path/filename)
    print("load checkpoint from ", filename)
    return checkpoint
    pass


def saveLoss(filename, iter, loss, task='train', **options):
    # save_file = save_output / "export.txt"
    with open(filename, "a") as myfile:
        myfile.write(task + " iter: " + str(iter) + ", ")
        myfile.write("loss: " + str(loss) + ", ")
        myfile.write(str(options))
        myfile.write("\n")

        # myfile.write("iter: " + str(iter) + '\n')
        # myfile.write("output pairs: " + str(count) + '\n')


def saveImg(img, filename):
    import cv2
    cv2.imwrite(filename, img)


def pltImshow(img):
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()


def loadConfig(filename):
    import yaml
    with open(filename, 'r') as f:
        config = yaml.load(f)
    return config


def append_csv(file='foo.csv', arr=[]):
    import csv
    # fields=['first','second','third']
    # pre = lambda i: ['{0:.3f}'.format(x) for x in i]
    with open(file, 'a') as f:
        writer = csv.writer(f)
        if type(arr[0]) is list:
            for a in arr:
                writer.writerow(a)
                # writer.writerow(pre(a))
                # print(pre(a))
        else:
            writer.writerow(arr)


def print_var(points):
    print("points: ", points.shape)
    print("points: ", points)
    pass


# from utils.losses import pts_to_bbox
def pts_to_bbox(points, patch_size):
    """
    input:
        points: (y, x)
    output:
        bbox: (x1, y1, x2, y2)
    """

    shift_l = (patch_size + 1) / 2
    shift_r = patch_size - shift_l
    pts_l = points - shift_l
    pts_r = points + shift_r + 1
    bbox = torch.stack((pts_l[:, 1], pts_l[:, 0], pts_r[:, 1], pts_r[:, 0]), dim=1)
    return bbox
    pass


def _roi_pool(pred_heatmap, rois, patch_size=8):
    from torchvision.ops import roi_pool
    patches = roi_pool(pred_heatmap, rois.float(), (patch_size, patch_size), spatial_scale=1.0)
    return patches
    pass


# from utils.losses import norm_patches
def norm_patches(patches):
    patch_size = patches.shape[-1]
    patches = patches.view(-1, 1, patch_size * patch_size)
    d = torch.sum(patches, dim=-1).unsqueeze(-1) + 1e-6
    patches = patches / d
    patches = patches.view(-1, 1, patch_size, patch_size)
    # print("patches: ", patches.shape)
    return patches


# from utils.losses import extract_patch_from_points
def extract_patch_from_points(heatmap, points, patch_size=5):
    """
    this function works in numpy
    """
    import numpy as np
    from util import toNumpy
    # numpy
    if type(heatmap) is torch.Tensor:
        heatmap = toNumpy(heatmap)
    heatmap = heatmap.squeeze()  # [H, W]
    # padding
    pad_size = int(patch_size / 2)
    heatmap = np.pad(heatmap, pad_size, 'constant')
    # crop it
    patches = []
    ext = lambda img, pnt, wid: img[pnt[1]:pnt[1] + wid, pnt[0]:pnt[0] + wid]
    # print("heatmap: ", heatmap.shape)
    for i in range(points.shape[0]):
        # print("point: ", points[i,:])
        patch = ext(heatmap, points[i, :].astype(int), patch_size)
        # print("patch: ", patch.shape)
        patches.append(patch)

        # if i > 10: break
    # extract points
    return patches


# from utils.losses import extract_patches
def extract_patches(label_idx, image, patch_size=7):
    """
    return:
        patches: tensor [N, 1, patch, patch]
    """
    rois = pts_to_bbox(label_idx[:, 2:], patch_size).long()
    # filter out??
    rois = torch.cat((label_idx[:, :1], rois), dim=1)
    # print_var(rois)
    # print_var(image)
    patches = _roi_pool(image, rois, patch_size=patch_size)
    return patches


# from utils.losses import points_to_4d
def points_to_4d(points):
    """
    input:
        points: tensor [N, 2] check(y, x)
    """
    num_of_points = points.shape[0]
    cols = torch.zeros(num_of_points, 1).float()
    points = torch.cat((cols, cols, points.float()), dim=1)
    return points


# from utils.losses import soft_argmax_2d
def soft_argmax_2d(patches, normalized_coordinates=True):
    """
    params:
        patches: (B, N, H, W)
    return:
        coor: (B, N, 2)  (x, y)

    """
    import torchgeometry as tgm
    m = tgm.contrib.SpatialSoftArgmax2d(normalized_coordinates=normalized_coordinates)
    coords = m(patches)  # 1x4x2
    return coords


## log on patches
# from utils.losses import do_log
def do_log(patches):
    patches[patches < 0] = 1e-6
    patches_log = torch.log(patches)
    return patches_log


def squeezeToNumpy(tensor_arr):
    return tensor_arr.detach().cpu().numpy().squeeze()


# from utils.losses import subpixel_loss
def subpixel_loss(labels_2D, labels_res, pred_heatmap, patch_size=7):
    """
    input:
        (tensor should be in GPU)
        labels_2D: tensor [batch, 1, H, W]
        labels_res: tensor [batch, 2, H, W]
        pred_heatmap: tensor [batch, 1, H, W]

    return:
        loss: sum of all losses
    """

    # soft argmax
    def _soft_argmax(patches):
        from models.SubpixelNet import SubpixelNet as subpixNet
        dxdy = subpixNet.soft_argmax_2d(patches)  # tensor [B, N, patch, patch]
        dxdy = dxdy.squeeze(1)  # tensor [N, 2]
        return dxdy

    points = labels_2D[...].nonzero()
    num_points = points.shape[0]
    if num_points == 0:
        return 0

    labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
    rois = pts_to_bbox(points[:, 2:], patch_size)
    # filter out??
    rois = torch.cat((points[:, :1], rois), dim=1)
    points_res = labels_res[points[:, 0], points[:, 1], points[:, 2], points[:, 3], :]  # tensor [N, 2]
    # print_var(rois)
    # print_var(labels_res)
    # print_var(points)
    # print("points max: ", points.max(dim=0))
    # print_var(labels_2D)
    # print_var(points_res)

    patches = _roi_pool(pred_heatmap, rois, patch_size=patch_size)
    # get argsoft max
    dxdy = _soft_argmax(patches)

    loss = (points_res - dxdy)
    loss = torch.norm(loss, p=2, dim=-1)
    loss = loss.sum() / num_points
    # print("loss: ", loss)
    return loss


def subpixel_loss_no_argmax(labels_2D, labels_res, pred_heatmap, **options):
    # extract points
    points = labels_2D[...].nonzero()
    num_points = points.shape[0]
    if num_points == 0:
        return 0

    def residual_from_points(labels_res, points):
        # extract residuals
        labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
        points_res = labels_res[points[:, 0], points[:, 1], points[:, 2], points[:, 3], :]  # tensor [N, 2]
        return points_res

    points_res = residual_from_points(labels_res, points)
    # print_var(points_res)
    # extract predicted residuals
    pred_res = residual_from_points(pred_heatmap, points)
    # print_var(pred_res)

    # loss
    loss = (points_res - pred_res)
    loss = torch.norm(loss, p=2, dim=-1).mean()
    # loss = loss.sum()/num_points
    return loss
    pass


def sample_homography(inv_scale=3):
  corner_img = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])
  # offset_r = 1 - 1/inv_scale
  # img_offset = np.array([(-1, -1), (-1, offset_r), (offset_r, -1), (offset_r, offset_r)])
  img_offset = corner_img
  corner_map = (np.random.rand(4,2)-0.5)*2/(inv_scale + 0.01) + img_offset
  matrix = cv2.getPerspectiveTransform(np.float32(corner_img), np.float32(corner_map))
  return matrix


def sample_homographies(batch_size=1, scale=10, device='cpu'):
    ## sample homography matrix
    # mat_H = [sample_homography(inv_scale=scale) for i in range(batch_size)]
    mat_H = [sample_homography(inv_scale=scale) for i in range(batch_size)]
    ##### debug
    # from utils.utils import sample_homo
    # mat_H = [sample_homo(image=np.zeros((1,1))) for i in range(batch_size)]

    # mat_H = [np.identity(3) for i in range(batch_size)]
    mat_H = np.stack(mat_H, axis=0)
    mat_H = torch.tensor(mat_H, dtype=torch.float32)
    mat_H = mat_H.to(device)

    mat_H_inv = torch.stack([torch.inverse(mat_H[i, :, :]) for i in range(batch_size)])
    mat_H_inv = torch.tensor(mat_H_inv, dtype=torch.float32)
    mat_H_inv = mat_H_inv.to(device)
    return mat_H, mat_H_inv


def warpLabels(pnts, homography, H, W):
    import torch
    """
    input:
        pnts: numpy
        homography: numpy
    output:
        warped_pnts: numpy
    """
    pnts = torch.tensor(pnts).long()
    homography = torch.tensor(homography, dtype=torch.float32)
    warped_pnts = warp_points(torch.stack((pnts[:, 0], pnts[:, 1]), dim=1),
                              homography)  # check the (x, y)
    warped_pnts = filter_points(warped_pnts, torch.tensor([W, H])).round().long()
    return warped_pnts.numpy()


def warp_points_np(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.
    """
    # expand points len to (x, y, 1)
    batch_size = homographies.shape[0]
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    # points = points.to(device)
    # homographies = homographies.(batch_size*3,3)
    # warped_points = homographies*points
    # warped_points = homographies@points.transpose(0,1)
    warped_points = np.tensordot(homographies, points.transpose(), axes=([2], [0]))
    # normalize the points
    warped_points = warped_points.reshape([batch_size, 3, -1])
    warped_points = warped_points.transpose([0, 2, 1])
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points


def homography_scaling(homography, H, W):
    trans = np.array([[2./W, 0., -1], [0., 2./H, -1], [0., 0., 1.]])
    homography = np.linalg.inv(trans) @ homography @ trans
    return homography


def homography_scaling_torch(homography, H, W):
    trans = torch.tensor([[2./W, 0., -1], [0., 2./H, -1], [0., 0., 1.]])
    homography = (trans.inverse() @ homography @ trans)
    return homography


def filter_points(points, shape, return_mask=False):
    ### check!
    points = points.float()
    shape = shape.float()
    mask = (points >= 0) * (points <= shape-1)
    mask = (torch.prod(mask, dim=-1) == 1)
    if return_mask:
        return points[mask], mask
    return points [mask]
    # return points [torch.prod(mask, dim=-1) == 1]


def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    # homographies = homographies.unsqueeze(0) if len(homographies.shape) == 2 else homographies
    batch_size = homographies.shape[0]
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    homographies = homographies.view(batch_size*3,3)
    # warped_points = homographies*points
    # points = points.double()
    warped_points = homographies@points.transpose(0,1)
    # warped_points = np.tensordot(homographies, points.transpose(), axes=([2], [0]))
    # normalize the points
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0,:,:] if no_batches else warped_points


# from utils.utils import inv_warp_image_batch
def inv_warp_image_batch(img, mat_homo_inv, device='cpu', mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    '''
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)

    Batch, channel, H, W = img.shape
    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H)), dim=2)
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv, device)
    src_pixel_coords = src_pixel_coords.view([Batch, H, W, 2])
    src_pixel_coords = src_pixel_coords.float()

    warped_img = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)
    return warped_img


def inv_warp_image(img, mat_homo_inv, device='cpu', mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [H, W]
    '''
    warped_img = inv_warp_image_batch(img, mat_homo_inv, device, mode)
    return warped_img.squeeze()


def labels2Dto3D(labels, cell_size, add_dustbin=True):
    '''
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    '''
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    # labels = space2depth(labels).squeeze(0)
    labels = space2depth(labels)
    # labels = labels.view(batch_size, H, 1, W, 1)
    # labels = labels.view(batch_size, Hc, cell_size, Wc, cell_size)
    # labels = labels.transpose(1, 2).transpose(3, 4).transpose(2, 3)
    # labels = labels.reshape(batch_size, 1, cell_size ** 2, Hc, Wc)
    # labels = labels.view(batch_size, cell_size ** 2, Hc, Wc)
    if add_dustbin:
        dustbin = labels.sum(dim=1)
        dustbin = 1 - dustbin
        dustbin[dustbin < 1.] = 0
        # print('dust: ', dustbin.shape)
        # labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        ## norm
        dn = labels.sum(dim=1)
        labels = labels.div(torch.unsqueeze(dn, 1))
    return labels


def labels2Dto3D_flattened(labels, cell_size):
    '''
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    '''
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    # labels = space2depth(labels).squeeze(0)
    labels = space2depth(labels)
    # print("labels in 2Dto3D: ", labels.shape)
    # labels = labels.view(batch_size, H, 1, W, 1)
    # labels = labels.view(batch_size, Hc, cell_size, Wc, cell_size)
    # labels = labels.transpose(1, 2).transpose(3, 4).transpose(2, 3)
    # labels = labels.reshape(batch_size, 1, cell_size ** 2, Hc, Wc)
    # labels = labels.view(batch_size, cell_size ** 2, Hc, Wc)

    dustbin = torch.ones((batch_size, 1, Hc, Wc)).cuda()
    # labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
    labels = torch.cat((labels*2, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
    labels = torch.argmax(labels, dim=1)
    return labels


def old_flatten64to1(semi, tensor=False):
    '''
    Flatten 3D np array to 2D

    :param semi:
        np [64 x Hc x Wc]
        or
        tensor (batch_size, 65, Hc, Wc)
    :return:
        flattened map
        np [1 x Hc*8 x Wc*8]
        or
        tensor (batch_size, 1, Hc*8, Wc*8)
    '''
    if tensor:
        is_batch = len(semi.size()) == 4
        if not is_batch:
            semi = semi.unsqueeze_(0)
        Hc, Wc = semi.size()[2], semi.size()[3]
        cell = 8
        semi.transpose_(1, 2)
        semi.transpose_(2, 3)
        semi = semi.view(-1, Hc, Wc, cell, cell)
        semi.transpose_(2, 3)
        semi = semi.contiguous()
        semi = semi.view(-1, 1, Hc * cell, Wc * cell)
        heatmap = semi
        if not is_batch:
            heatmap = heatmap.squeeze_(0)
    else:
        Hc, Wc = semi.shape[1], semi.shape[2]
        cell = 8
        semi = semi.transpose(1, 2, 0)
        heatmap = np.reshape(semi, [Hc, Wc, cell, cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        # heatmap = np.transpose(heatmap, [2, 0, 3, 1])
        heatmap = np.reshape(heatmap, [Hc * cell, Wc * cell])
        heatmap = heatmap[np.newaxis, :, :]
    return heatmap


def flattenDetection(semi, tensor=False):
    '''
    Flatten detection output

    :param semi:
        output from detector head
        tensor [65, Hc, Wc]
        :or
        tensor (batch_size, 65, Hc, Wc)

    :return:
        3D heatmap
        np (1, H, C)
        :or
        tensor (batch_size, 65, Hc, Wc)

    '''
    batch = False
    if len(semi.shape) == 4:
        batch = True
        batch_size = semi.shape[0]
    # if tensor:
    #     semi.exp_()
    #     d = semi.sum(dim=1) + 0.00001
    #     d = d.view(d.shape[0], 1, d.shape[1], d.shape[2])
    #     semi = semi / d  # how to /(64,15,20)

    #     nodust = semi[:, :-1, :, :]
    #     heatmap = flatten64to1(nodust, tensor=tensor)
    # else:
    # Convert pytorch -> numpy.
    # --- Process points.
    # dense = nn.functional.softmax(semi, dim=0) # [65, Hc, Wc]
    if batch:
        dense = nn.functional.softmax(semi, dim=1) # [batch, 65, Hc, Wc]
        # Remove dustbin.
        nodust = dense[:, :-1, :, :]
    else:
        dense = nn.functional.softmax(semi, dim=0) # [65, Hc, Wc]
        nodust = dense[:-1, :, :].unsqueeze(0)
    # Reshape to get full resolution heatmap.
    # heatmap = flatten64to1(nodust, tensor=True) # [1, H, W]
    depth2space = DepthToSpace(8)
    heatmap = depth2space(nodust)
    heatmap = heatmap.squeeze(0) if not batch else heatmap
    return heatmap


def sample_homo(image):
    import tensorflow as tf
    from utils.homographies import sample_homography
    H = sample_homography(tf.shape(image)[:2])
    with tf.Session():
        H_ = H.eval()
    H_ = np.concatenate((H_, np.array([1])[:, np.newaxis]), axis=1)
    # warped_im = tf.contrib.image.transform(image, H, interpolation="BILINEAR")
    mat = np.reshape(H_, (3, 3))
    # for i in range(batch):
    #     np.stack()
    return mat


def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist):
    '''
    :param self:
    :param heatmap:
        np (H, W)
    :return:
    '''

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


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    # requires https://github.com/open-mmlab/mmdetection.
    # Warning : BUILD FROM SOURCE using command MMCV_WITH_OPS=1 pip install -e
    # from mmcv.ops import nms as nms_mmdet
    from torchvision.ops import nms

    """Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.
    Arguments:
    prob: the probability heatmap, with shape `[H, W]`.
    size: a scalar, the size of the bouding boxes.
    iou: a scalar, the IoU overlap threshold.
    min_prob: a threshold under which all probabilities are discarded before NMS.
    keep_top_k: an integer, the number of top scores to keep.
    """
    pts = torch.nonzero(prob > min_prob).float() # [N, 2]
    prob_nms = torch.zeros_like(prob)
    if pts.nelement() == 0:
        return prob_nms
    size = torch.tensor(size/2.).cuda()
    boxes = torch.cat([pts-size, pts+size], dim=1) # [N, 4]
    scores = prob[pts[:, 0].long(), pts[:, 1].long()]
    if keep_top_k != 0:
        indices = nms(boxes, scores, iou)
    else:
        raise NotImplementedError
        # indices, _ = nms(boxes, scores, iou, boxes.size()[0])
        # print("boxes: ", boxes.shape)
        # print("scores: ", scores.shape)
        # proposals = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)
        # dets, indices = nms_mmdet(proposals, iou)
        # indices = indices.long()

        # indices = box_nms_retinaNet(boxes, scores, iou)
    pts = torch.index_select(pts, 0, indices)
    scores = torch.index_select(scores, 0, indices)
    prob_nms[pts[:, 0].long(), pts[:, 1].long()] = scores
    return prob_nms


def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds


def compute_valid_mask(image_shape, inv_homography, device='cpu', erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        `erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """
    # mask = H_transform(tf.ones(image_shape), homography, interpolation='NEAREST')
    # mask = H_transform(tf.ones(image_shape), homography, interpolation='NEAREST')
    if inv_homography.dim() == 2:
        inv_homography = inv_homography.view(-1, 3, 3)
    batch_size = inv_homography.shape[0]
    mask = torch.ones(batch_size, 1, image_shape[0], image_shape[1]).to(device)
    mask = inv_warp_image_batch(mask, inv_homography, device=device, mode='nearest')
    mask = mask.view(batch_size, image_shape[0], image_shape[1])
    mask = mask.cpu().numpy()
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)

    return torch.tensor(mask).to(device)


def normPts(pts, shape):
    """
    normalize pts to [-1, 1]
    :param pts:
        tensor (y, x)
    :param shape:
        tensor shape (y, x)
    :return:
    """
    pts = pts/shape*2 - 1
    return pts


def denormPts(pts, shape):
    """
    denormalize pts back to H, W
    :param pts:
        tensor (y, x)
    :param shape:
        numpy (y, x)
    :return:
    """
    pts = (pts+1)*shape/2
    return pts


def descriptor_loss(descriptors, descriptors_warped, homographies, mask_valid=None,
                    cell_size=8, lamda_d=250, device='cpu', descriptor_dist=4, **config):

    '''
    Compute descriptor loss from descriptors_warped and given homographies

    :param descriptors:
        Output from descriptor head
        tensor [batch_size, descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [batch_size, descriptors, Hc, Wc]
    :param homographies:
        known homographies
    :param cell_size:
        8
    :param device:
        gpu or cpu
    :param config:
    :return:
        loss, and other tensors for visualization
    '''

    # put to gpu
    homographies = homographies.to(device)
    # config
    lamda_d = lamda_d # 250
    margin_pos = 1
    margin_neg = 0.2
    batch_size, Hc, Wc = descriptors.shape[0], descriptors.shape[2], descriptors.shape[3]
    #####
    # H, W = Hc.numpy().astype(int) * cell_size, Wc.numpy().astype(int) * cell_size
    H, W = Hc * cell_size, Wc * cell_size
    #####
    with torch.no_grad():
        # shape = torch.tensor(list(descriptors.shape[2:]))*torch.tensor([cell_size, cell_size]).type(torch.FloatTensor).to(device)
        shape = torch.tensor([H, W]).type(torch.FloatTensor).to(device)
        # compute the center pixel of every cell in the image

        coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2)
        coor_cells = coor_cells.type(torch.FloatTensor).to(device)
        coor_cells = coor_cells * cell_size + cell_size // 2
        ## coord_cells is now a grid containing the coordinates of the Hc x Wc
        ## center pixels of the 8x8 cells of the image

        # coor_cells = coor_cells.view([-1, Hc, Wc, 1, 1, 2])
        coor_cells = coor_cells.view([-1, 1, 1, Hc, Wc, 2])  # be careful of the order
        # warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
        warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
        warped_coor_cells = torch.stack((warped_coor_cells[:,1], warped_coor_cells[:,0]), dim=1) # (y, x) to (x, y)
        warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

        warped_coor_cells = torch.stack((warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2)  # (batch, x, y) to (batch, y, x)

        shape_cell = torch.tensor([H//cell_size, W//cell_size]).type(torch.FloatTensor).to(device)
        # warped_coor_mask = denormPts(warped_coor_cells, shape_cell)

        warped_coor_cells = denormPts(warped_coor_cells, shape)
        # warped_coor_cells = warped_coor_cells.view([-1, 1, 1, Hc, Wc, 2])
        warped_coor_cells = warped_coor_cells.view([-1, Hc, Wc, 1, 1, 2])
    #     print("warped_coor_cells: ", warped_coor_cells.shape)
        # compute the pairwise distance
        cell_distances = coor_cells - warped_coor_cells
        cell_distances = torch.norm(cell_distances, dim=-1)
        ##### check
    #     print("descriptor_dist: ", descriptor_dist)
        mask = cell_distances <= descriptor_dist # 0.5 # trick

        mask = mask.type(torch.FloatTensor).to(device)

    # compute the pairwise dot product between descriptors: d^t * d
    descriptors = descriptors.transpose(1, 2).transpose(2, 3)
    descriptors = descriptors.view((batch_size, Hc, Wc, 1, 1, -1))
    descriptors_warped = descriptors_warped.transpose(1, 2).transpose(2, 3)
    descriptors_warped = descriptors_warped.view((batch_size, 1, 1, Hc, Wc, -1))
    dot_product_desc = descriptors * descriptors_warped
    dot_product_desc = dot_product_desc.sum(dim=-1)
    ## dot_product_desc.shape = [batch_size, Hc, Wc, Hc, Wc, desc_len]

    # hinge loss
    positive_dist = torch.max(margin_pos - dot_product_desc, torch.tensor(0.).to(device))
    # positive_dist[positive_dist < 0] = 0
    negative_dist = torch.max(dot_product_desc - margin_neg, torch.tensor(0.).to(device))
    # negative_dist[neative_dist < 0] = 0
    # sum of the dimension

    if mask_valid is None:
        # mask_valid = torch.ones_like(mask)
        mask_valid = torch.ones(batch_size, 1, Hc*cell_size, Wc*cell_size)
    mask_valid = mask_valid.view(batch_size, 1, 1, mask_valid.shape[2], mask_valid.shape[3])

    loss_desc = lamda_d * mask * positive_dist + (1 - mask) * negative_dist
    loss_desc = loss_desc * mask_valid
        # mask_validg = torch.ones_like(mask)
    ##### bug in normalization
    normalization = (batch_size * (mask_valid.sum()+1) * Hc * Wc)
    pos_sum = (lamda_d * mask * positive_dist/normalization).sum()
    neg_sum = ((1 - mask) * negative_dist/normalization).sum()
    loss_desc = loss_desc.sum() / normalization
    # loss_desc = loss_desc.sum() / (batch_size * Hc * Wc)
    # return loss_desc, mask, mask_valid, positive_dist, negative_dist
    return loss_desc, mask, pos_sum, neg_sum


def sumto2D(ndtensor):
    # input tensor: [batch_size, Hc, Wc, Hc, Wc]
    # output tensor: [batch_size, Hc, Wc]
    return ndtensor.sum(dim=1).sum(dim=1)


def mAP(pred_batch, labels_batch):
    pass


def precisionRecall_torch(pred, labels):
    offset = 10**-6
    assert pred.size() == labels.size(), 'Sizes of pred, labels should match when you get the precision/recall!'
    precision = torch.sum(pred*labels) / (torch.sum(pred)+ offset)
    recall = torch.sum(pred*labels) / (torch.sum(labels) + offset)
    if precision.item() > 1.:
        print(pred)
        print(labels)
        import scipy.io.savemat as savemat
        savemat('pre_recall.mat', {'pred': pred, 'labels': labels})
    assert precision.item() <=1. and precision.item() >= 0.
    return {'precision': precision, 'recall': recall}


def precisionRecall(pred, labels, thd=None):
    offset = 10**-6
    if thd is None:
        precision = np.sum(pred*labels) / (np.sum(pred)+ offset)
        recall = np.sum(pred*labels) / (np.sum(labels) + offset)
    return {'precision': precision, 'recall': recall}


def getWriterPath(task='train', exper_name='', date=True):
    import datetime
    prefix = 'runs/'
    str_date_time = ''
    if exper_name != '':
        exper_name += '_'
    if date:
        str_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return prefix + task + '/' + exper_name + str_date_time


def crop_or_pad_choice(in_num_points, out_num_points, shuffle=False):
    # Adapted from https://github.com/haosulab/frustum_pointnet/blob/635c938f18b9ec1de2de717491fb217df84d2d93/fpointnet/data/datasets/utils.py
    """Crop or pad point cloud to a fixed number; return the indexes
    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order
    Returns:
        np.ndarray: output point cloud
        np.ndarray: index to choose input points
    """
    if shuffle:
        choice = np.random.permutation(in_num_points)
    else:
        choice = np.arange(in_num_points)
    assert out_num_points > 0, 'out_num_points = %d must be positive int!'%out_num_points
    if in_num_points >= out_num_points:
        choice = choice[:out_num_points]
    else:
        num_pad = out_num_points - in_num_points
        pad = np.random.choice(choice, num_pad, replace=True)
        choice = np.concatenate([choice, pad])
    return choice


def get_coor_cells(Hc, Wc, cell_size, device='cpu', uv=False):
    coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2)
    coor_cells = coor_cells.type(torch.FloatTensor).to(device)
    coor_cells = coor_cells.view(-1, 2)
    # change vu to uv
    if uv:
        coor_cells = torch.stack((coor_cells[:,1], coor_cells[:,0]), dim=1) # (y, x) to (x, y)

    return coor_cells.to(device)


def warp_coor_cells_with_homographies(coor_cells, homographies, uv=False, device='cpu'):

    # warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
    # warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
    warped_coor_cells = coor_cells
    if uv == False:
        warped_coor_cells = torch.stack((warped_coor_cells[:,1], warped_coor_cells[:,0]), dim=1) # (y, x) to (x, y)

    # print("homographies: ", homographies)
    warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

    if uv == False:
        warped_coor_cells = torch.stack((warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2)  # (batch, x, y) to (batch, y, x)

    # shape_cell = torch.tensor([H//cell_size, W//cell_size]).type(torch.FloatTensor).to(device)
    # warped_coor_mask = denormPts(warped_coor_cells, shape_cell)

    return warped_coor_cells


def create_non_matches(uv_a, uv_b_non_matches, multiplier):
    """
    Simple wrapper for repeated code
    :param uv_a:
    :type uv_a:
    :param uv_b_non_matches:
    :type uv_b_non_matches:
    :param multiplier:
    :type multiplier:
    :return:
    :rtype:
    """
    uv_a_long = (torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1),
                 torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1))

    uv_b_non_matches_long = (uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1))

    return uv_a_long, uv_b_non_matches_long


def descriptor_loss_sparse(descriptors, descriptors_warped, homographies, mask_valid=None,
                           cell_size=8, device='cpu', descriptor_dist=4, lamda_d=250,
                           num_matching_attempts=1000, num_masked_non_matches_per_match=10,
                           dist='cos', method='1d', **config):
    """
    consider batches of descriptors
    :param descriptors:
        Output from descriptor head
        tensor [descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [descriptors, Hc, Wc]
    """

    def uv_to_tuple(uv):
        return (uv[:, 0], uv[:, 1])

    def tuple_to_uv(uv_tuple):
        return torch.stack([uv_tuple[0], uv_tuple[1]])

    def tuple_to_1d(uv_tuple, W, uv=True):
        if uv:
            return uv_tuple[0] + uv_tuple[1]*W
        else:
            return uv_tuple[0]*W + uv_tuple[1]


    def uv_to_1d(points, W, uv=True):
        # assert points.dim == 2
        #     print("points: ", points[0])
        #     print("H: ", H)
        if uv:
            return points[..., 0] + points[..., 1]*W
        else:
            return points[..., 0]*W + points[..., 1]

    ## calculate matches loss
    def get_match_loss(image_a_pred, image_b_pred, matches_a, matches_b, dist='cos', method='1d'):
        match_loss, matches_a_descriptors, matches_b_descriptors = \
            PixelwiseContrastiveLoss.match_loss(image_a_pred, image_b_pred,
                matches_a, matches_b, dist=dist, method=method)
        return match_loss

    def get_non_matches_corr(img_b_shape, uv_a, uv_b_matches, num_masked_non_matches_per_match=10, device='cpu'):
        ## sample non matches
        uv_b_matches = uv_b_matches.squeeze()
        uv_b_matches_tuple = uv_to_tuple(uv_b_matches)
        uv_b_non_matches_tuple = create_non_correspondences(uv_b_matches_tuple,
                                        img_b_shape, num_non_matches_per_match=num_masked_non_matches_per_match,
                                        img_b_mask=None)

        ## create_non_correspondences
        #     print("img_b_shape ", img_b_shape)
        #     print("uv_b_matches ", uv_b_matches.shape)
        # print("uv_a: ", uv_to_tuple(uv_a))
        # print("uv_b_non_matches: ", uv_b_non_matches)
        #     print("uv_b_non_matches: ", tensorUv2tuple(uv_b_non_matches))
        uv_a_tuple, uv_b_non_matches_tuple = \
            create_non_matches(uv_to_tuple(uv_a), uv_b_non_matches_tuple, num_masked_non_matches_per_match)
        return uv_a_tuple, uv_b_non_matches_tuple

    def get_non_match_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_b, dist='cos'):
        ## non matches loss
        non_match_loss, num_hard_negatives, non_matches_a_descriptors, non_matches_b_descriptors = \
            PixelwiseContrastiveLoss.non_match_descriptor_loss(image_a_pred, image_b_pred,
                                                               non_matches_a.long().squeeze(),
                                                               non_matches_b.long().squeeze(),
                                                               M=0.2, invert=True, dist=dist)
        non_match_loss = non_match_loss.sum()/(num_hard_negatives + 1)
        return non_match_loss

    # ##### print configs
    # print("num_masked_non_matches_per_match: ", num_masked_non_matches_per_match)
    # print("num_matching_attempts: ", num_matching_attempts)
    # dist = 'cos'
    # print("method: ", method)

    Hc, Wc = descriptors.shape[1], descriptors.shape[2]
    img_shape = (Hc, Wc)
    # print("img_shape: ", img_shape)
    # img_shape_cpu = (Hc.to('cpu'), Wc.to('cpu'))

    # image_a_pred = descriptors.view(1, -1, Hc * Wc).transpose(1, 2)  # torch [batch_size, H*W, D]
    def descriptor_reshape(descriptors):
        descriptors = descriptors.view(-1, Hc * Wc).transpose(0, 1)  # torch [D, H, W] --> [H*W, d]
        descriptors = descriptors.unsqueeze(0)  # torch [1, H*W, D]
        return descriptors

    image_a_pred = descriptor_reshape(descriptors)  # torch [1, H*W, D]
    # print("image_a_pred: ", image_a_pred.shape)
    image_b_pred = descriptor_reshape(descriptors_warped)  # torch [batch_size, H*W, D]

    # matches
    uv_a = get_coor_cells(Hc, Wc, cell_size, uv=True, device='cpu')
    # print("uv_a: ", uv_a[0])

    homographies_H = scale_homography_torch(homographies, img_shape, shift=(-1, -1))

    # print("experiment inverse homographies")
    # homographies_H = torch.stack([torch.inverse(H) for H in homographies_H])
    # print("homographies_H: ", homographies_H.shape)
    # homographies_H = torch.inverse(homographies_H)


    uv_b_matches = warp_coor_cells_with_homographies(uv_a, homographies_H.to('cpu'), uv=True, device='cpu')
    #
    # print("uv_b_matches before round: ", uv_b_matches[0])

    uv_b_matches.round_()
    # print("uv_b_matches after round: ", uv_b_matches[0])
    uv_b_matches = uv_b_matches.squeeze(0)


    # filtering out of range points
    # choice = crop_or_pad_choice(x_all.shape[0], self.sift_num, shuffle=True)

    uv_b_matches, mask = filter_points(uv_b_matches, torch.tensor([Wc, Hc]).to(device='cpu'), return_mask=True)
    # print ("pos mask sum: ", mask.sum())
    uv_a = uv_a[mask]

    # crop to the same length
    shuffle = True
    if not shuffle: print("shuffle: ", shuffle)
    choice = crop_or_pad_choice(uv_b_matches.shape[0], num_matching_attempts, shuffle=shuffle)
    choice = torch.tensor(choice).type(torch.int64)
    uv_a = uv_a[choice]
    uv_b_matches = uv_b_matches[choice]

    if method == '2d':
        matches_a = normPts(uv_a, torch.tensor([Wc, Hc]).float()) # [u, v]
        matches_b = normPts(uv_b_matches, torch.tensor([Wc, Hc]).float())
    else:
        matches_a = uv_to_1d(uv_a, Wc)
        matches_b = uv_to_1d(uv_b_matches, Wc)

    # print("matches_a: ", matches_a.shape)
    # print("matches_b: ", matches_b.shape)
    # print("matches_b max: ", matches_b.max())

    if method == '2d':
        match_loss = get_match_loss(descriptors, descriptors_warped, matches_a.to(device),
            matches_b.to(device), dist=dist, method='2d')
    else:
        match_loss = get_match_loss(image_a_pred, image_b_pred,
            matches_a.long().to(device), matches_b.long().to(device), dist=dist)

    # non matches

    # get non matches correspondence
    uv_a_tuple, uv_b_non_matches_tuple = get_non_matches_corr(img_shape,
                                            uv_a, uv_b_matches,
                                            num_masked_non_matches_per_match=num_masked_non_matches_per_match)

    non_matches_a = tuple_to_1d(uv_a_tuple, Wc)
    non_matches_b = tuple_to_1d(uv_b_non_matches_tuple, Wc)

    # print("non_matches_a: ", non_matches_a)
    # print("non_matches_b: ", non_matches_b)

    non_match_loss = get_non_match_loss(image_a_pred, image_b_pred, non_matches_a.to(device),
                                        non_matches_b.to(device), dist=dist)
    # non_match_loss = non_match_loss.mean()

    loss = lamda_d * match_loss + non_match_loss
    return loss, lamda_d * match_loss, non_match_loss
    pass


def batch_descriptor_loss_sparse(descriptors, descriptors_warped, homographies, **options):
    loss = []
    pos_loss = []
    neg_loss = []
    batch_size = descriptors.shape[0]
    for i in range(batch_size):
        losses = descriptor_loss_sparse(descriptors[i], descriptors_warped[i],
                    # torch.tensor(homographies[i], dtype=torch.float32), **options)
                    homographies[i].type(torch.float32), **options)
        loss.append(losses[0])
        pos_loss.append(losses[1])
        neg_loss.append(losses[2])
    loss, pos_loss, neg_loss = torch.stack(loss), torch.stack(pos_loss), torch.stack(neg_loss)
    return loss.mean(), None, pos_loss.mean(), neg_loss.mean()

def sample_homography_np(
        shape, shift=0, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=pi/2,
        allow_artifacts=False, translation_overflow=0.):
    """Sample a random valid homography.

    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """

    # print("debugging")


    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio], [patch_ratio, 0]])

    from numpy.random import normal
    from numpy.random import uniform
    from scipy.stats import truncnorm

    # Random perspective and affine perturbations
    # lower, upper = 0, 2
    std_trunc = 2

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        # perspective_displacement = tf.truncated_normal([1], 0., perspective_amplitude_y/2)
        # perspective_displacement = normal(0., perspective_amplitude_y/2, 1)
        perspective_displacement = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y/2).rvs(1)
        # h_displacement_left = normal(0., perspective_amplitude_x/2, 1)
        h_displacement_left = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        # h_displacement_right = normal(0., perspective_amplitude_x/2, 1)
        h_displacement_right = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = truncnorm(-1*std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)

        # scales = np.concatenate( (np.ones((n_scales,1)), scales[:,np.newaxis]), axis=1)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            # valid = np.where((scaled >= 0.) * (scaled < 1.))
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx,:,:]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx,:,:]
        # idx = valid[tf.random_uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)]
        # pts2 = rotated[idx]

    # Rescale to actual size
    shape = shape[::-1]  # different convention [y, x]
    pts1 *= shape[np.newaxis,:]
    pts2 *= shape[np.newaxis,:]

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    # a_mat = tf.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    # p_mat = tf.transpose(tf.stack(
    #     [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))
    # homography = tf.transpose(tf.matrix_solve_ls(a_mat, p_mat, fast=True))
    homography = cv2.getPerspectiveTransform(np.float32(pts1+shift), np.float32(pts2+shift))
    return homography


class ImgAugTransform:
    def __init__(self, **config):
        from numpy.random import uniform
        from numpy.random import randint

        ## old photometric
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.Sometimes(0.25,
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5),
                          )
        ])

        if config['photometric']['enable']:
            params = config['photometric']['params']
            aug_all = []
            if params.get('random_brightness', False):
                change = params['random_brightness']['max_abs_change']
                aug = iaa.Add((-change, change))
                #                 aug_all.append(aug)
                aug_all.append(aug)
            # if params['random_contrast']:
            if params.get('random_contrast', False):
                change = params['random_contrast']['strength_range']
                aug = iaa.LinearContrast((change[0], change[1]))
                aug_all.append(aug)
            # if params['additive_gaussian_noise']:
            if params.get('additive_gaussian_noise', False):
                change = params['additive_gaussian_noise']['stddev_range']
                aug = iaa.AdditiveGaussianNoise(scale=(change[0], change[1]))
                aug_all.append(aug)
            # if params['additive_speckle_noise']:
            if params.get('additive_speckle_noise', False):
                change = params['additive_speckle_noise']['prob_range']
                # aug = iaa.Dropout(p=(change[0], change[1]))
                aug = iaa.ImpulseNoise(p=(change[0], change[1]))
                aug_all.append(aug)
            # if params['motion_blur']:
            if params.get('motion_blur', False):
                change = params['motion_blur']['max_kernel_size']
                if change > 3:
                    change = randint(3, change)
                elif change == 3:
                    aug = iaa.Sometimes(0.5, iaa.MotionBlur(change))
                aug_all.append(aug)

            if params.get('GaussianBlur', False):
                change = params['GaussianBlur']['sigma']
                aug = iaa.GaussianBlur(sigma=(change))
                aug_all.append(aug)

            self.aug = iaa.Sequential(aug_all)


        else:
            self.aug = iaa.Sequential([
                iaa.Noop(),
            ])

    def __call__(self, img):
        img = np.array(img)
        img = (img * 255).astype(np.uint8)
        img = self.aug.augment_image(img)
        img = img.astype(np.float32) / 255
        return img


class customizedTransform:
    def __init__(self):
        pass

    def additive_shade(self, image, nb_ellipses=20, transparency_range=[-0.5, 0.8],
                       kernel_size_range=[250, 350]):
        def _py_additive_shade(img):
            min_dim = min(img.shape[:2]) / 4
            mask = np.zeros(img.shape[:2], np.uint8)
            for i in range(nb_ellipses):
                ax = int(max(np.random.rand() * min_dim, min_dim / 5))
                ay = int(max(np.random.rand() * min_dim, min_dim / 5))
                max_rad = max(ax, ay)
                x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
                y = np.random.randint(max_rad, img.shape[0] - max_rad)
                angle = np.random.rand() * 90
                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

            transparency = np.random.uniform(*transparency_range)
            kernel_size = np.random.randint(*kernel_size_range)
            if (kernel_size % 2) == 0:  # kernel_size has to be odd
                kernel_size += 1
            mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
#             shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.)
            shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.)
            return np.clip(shaded, 0, 255)

        shaded = _py_additive_shade(image)
        return shaded

    def __call__(self, img, **config):
        if config['photometric']['params']['additive_shade']:
            params = config['photometric']['params']
            img = self.additive_shade(img * 255, **params['additive_shade'])
        return img / 255

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

def scale_homography_torch(H, shape, shift=(-1,-1), dtype=torch.float32):
    height, width = shape[0], shape[1]
    trans = torch.tensor([[2./width, 0., shift[0]], [0., 2./height, shift[1]], [0., 0., 1.]], dtype=dtype)
    # print("torch.inverse(trans) ", torch.inverse(trans))
    # print("H: ", H)
    H_tf = torch.inverse(trans) @ H @ trans
    return H_tf


class PixelwiseContrastiveLoss(object):

    def __init__(self, image_shape, config=None):
        self.type = "pixelwise_contrastive"
        self.image_width = image_shape[1]
        self.image_height = image_shape[0]

        assert config is not None
        self._config = config

        self._debug_data = dict()

        self._debug = False

    @property
    def debug(self):
        return self._debug

    @property
    def config(self):
        return self._config

    @debug.setter
    def debug(self, value):
        self._debug = value

    @property
    def debug_data(self):
        return self._debug_data

    def get_loss_matched_and_non_matched_with_l2(self, image_a_pred, image_b_pred, matches_a, matches_b, non_matches_a,
                                                 non_matches_b,
                                                 M_descriptor=None, M_pixel=None, non_match_loss_weight=1.0,
                                                 use_l2_pixel_loss=None):
        """
        Computes the loss function

        DCN = Dense Correspondence Network
        num_images = number of images in this batch
        num_matches = number of matches
        num_non_matches = number of non-matches
        W = image width
        H = image height
        D = descriptor dimension


        match_loss = 1/num_matches \sum_{num_matches} ||descriptor_a - descriptor_b||_2^2
        non_match_loss = 1/num_non_matches \sum_{num_non_matches} max(0, M_margin - ||descriptor_a - descriptor_b||_2)^2

        loss = match_loss + non_match_loss

        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_a
        :type matches_b:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a
        :type non_matches_b:
        :return: loss, match_loss, non_match_loss
        :rtype: torch.Variable(torch.FloatTensor) each of shape torch.Size([1])
        """

        PCL = PixelwiseContrastiveLoss

        if M_descriptor is None:
            M_descriptor = self._config["M_descriptor"]

        if M_pixel is None:
            M_pixel = self._config["M_pixel"]

        if use_l2_pixel_loss is None:
            use_l2_pixel_loss = self._config['use_l2_pixel_loss_on_masked_non_matches']

        match_loss, _, _ = PCL.match_loss(image_a_pred, image_b_pred, matches_a, matches_b)

        if use_l2_pixel_loss:
            non_match_loss, num_hard_negatives = \
                self.non_match_loss_with_l2_pixel_norm(image_a_pred, image_b_pred, matches_b,
                                                       non_matches_a, non_matches_b,
                                                       M_descriptor=M_descriptor,
                                                       M_pixel=M_pixel)
        else:
            # version with no l2 pixel term
            non_match_loss, num_hard_negatives = self.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                                     non_matches_a, non_matches_b,
                                                                                     M_descriptor=M_descriptor)

        return match_loss, non_match_loss, num_hard_negatives

    @staticmethod
    def get_triplet_loss(image_a_pred, image_b_pred, matches_a, matches_b, non_matches_a, non_matches_b, alpha):
        """
        Computes the loss function

        \sum_{triplets} ||D(I_a, u_a, I_b, u_{b,match})||_2^2 - ||D(I_a, u_a, I_b, u_{b,non-match)||_2^2 + alpha

        """
        num_matches = matches_a.size()[0]
        num_non_matches = non_matches_a.size()[0]
        multiplier = num_non_matches / num_matches

        ## non_matches_a is already replicated up to be the right size
        ## non_matches_b is also that side
        ## matches_a is just a smaller version of non_matches_a
        ## matches_b is the only thing that needs to be replicated up in size

        matches_b_long = torch.t(matches_b.repeat(multiplier, 1)).contiguous().view(-1)

        matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a)
        matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b_long)
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b)

        triplet_losses = (matches_a_descriptors - matches_b_descriptors).pow(2) - (
                    matches_a_descriptors - non_matches_b_descriptors).pow(2) + alpha
        triplet_loss = 1.0 / num_non_matches * torch.clamp(triplet_losses, min=0).sum()

        return triplet_loss

    @staticmethod
    def match_loss(image_a_pred, image_b_pred, matches_a, matches_b, M=1.0,
                   dist='euclidean', method='1d'):  # dist = 'cos'
        """
        Computes the match loss given by

        1/num_matches * \sum_{matches} ||D(I_a, u_a, I_b, u_b)||_2^2

        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_b

        :return: match_loss, matches_a_descriptors, matches_b_descriptors
        :rtype: torch.Variable(),

        matches_a_descriptors is torch.FloatTensor with shape torch.Shape([num_matches, descriptor_dimension])
        """
        if method == '2d':
            import torch.nn.functional as F
            num_matches = matches_a.size()[0]
            mode = 'bilinear'

            def sampleDescriptors(image_a_pred, matches_a, mode, norm=False):
                image_a_pred = image_a_pred.unsqueeze(0)  # torch [1, D, H, W]
                matches_a.unsqueeze_(0).unsqueeze_(2)
                matches_a_descriptors = F.grid_sample(image_a_pred, matches_a, mode=mode, align_corners=True)
                matches_a_descriptors = matches_a_descriptors.squeeze().transpose(0, 1)

                # print("image_a_pred: ", image_a_pred.shape)
                # print("matches_a: ", matches_a.shape)
                # print("matches_a: ", matches_a)
                # print("matches_a_descriptors: ", matches_a_descriptors)
                if norm:
                    dn = torch.norm(matches_a_descriptors, p=2, dim=1)  # Compute the norm of b_descriptors
                    matches_a_descriptors = matches_a_descriptors.div(
                        torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
                return matches_a_descriptors

            # image_b_pred = image_b_pred.unsqueeze(0) # torch [1, D, H, W]
            # matches_b.unsqueeze_(0).unsqueeze_(2)
            # matches_b_descriptors = F.grid_sample(image_b_pred, matches_b, mode=mode)
            # matches_a_descriptors = matches_a_descriptors.squeeze().transpose(0,1)
            norm = False
            matches_a_descriptors = sampleDescriptors(image_a_pred, matches_a, mode, norm=norm)
            matches_b_descriptors = sampleDescriptors(image_b_pred, matches_b, mode, norm=norm)
        else:
            num_matches = matches_a.size()[0]
            matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
            matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)

        # crazily enough, if there is only one element to index_select into
        # above, then the first dimension is collapsed down, and we end up
        # with shape [D,], where we want [1,D]
        # this unsqueeze fixes that case
        if len(matches_a) == 1:
            matches_a_descriptors = matches_a_descriptors.unsqueeze(0)
            matches_b_descriptors = matches_b_descriptors.unsqueeze(0)

        if dist == 'cos':
            # print("dot product: ", (matches_a_descriptors * matches_b_descriptors).shape)
            match_loss = torch.clamp(M - (matches_a_descriptors * matches_b_descriptors).sum(dim=-1), min=0)
            match_loss = 1.0 / num_matches * match_loss.sum()
        else:
            match_loss = 1.0 / num_matches * (matches_a_descriptors - matches_b_descriptors).pow(2).sum()

        return match_loss, matches_a_descriptors, matches_b_descriptors

    @staticmethod
    def non_match_descriptor_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_b, M=0.5, invert=False,
                                  dist='euclidear'):
        """
        Computes the max(0, M - D(I_a,I_b,u_a,u_b))^2 term

        This is effectively:       "a and b should be AT LEAST M away from each other"
        With invert=True, this is: "a and b should be AT MOST  M away from each other"

         :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a
        :param M: the margin
        :type M: float
        :return: torch.FloatTensor with shape torch.Shape([num_non_matches])
        :rtype:
        """

        non_matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a).squeeze()
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b).squeeze()

        # crazily enough, if there is only one element to index_select into
        # above, then the first dimension is collapsed down, and we end up
        # with shape [D,], where we want [1,D]
        # this unsqueeze fixes that case
        if len(non_matches_a) == 1:
            non_matches_a_descriptors = non_matches_a_descriptors.unsqueeze(0)
            non_matches_b_descriptors = non_matches_b_descriptors.unsqueeze(0)

        norm_degree = 2
        if dist == 'cos':
            non_match_loss = (non_matches_a_descriptors * non_matches_b_descriptors).sum(dim=-1)
        else:
            non_match_loss = (non_matches_a_descriptors - non_matches_b_descriptors).norm(norm_degree, 1)
        if not invert:
            non_match_loss = torch.clamp(M - non_match_loss, min=0).pow(2)
        else:
            if dist == 'cos':
                non_match_loss = torch.clamp(non_match_loss - M, min=0)
            else:
                non_match_loss = torch.clamp(non_match_loss - M, min=0).pow(2)

        hard_negative_idxs = torch.nonzero(non_match_loss)
        num_hard_negatives = len(hard_negative_idxs)

        return non_match_loss, num_hard_negatives, non_matches_a_descriptors, non_matches_b_descriptors

    def non_match_loss_with_l2_pixel_norm(self, image_a_pred, image_b_pred, matches_b,
                                          non_matches_a, non_matches_b, M_descriptor=0.5,
                                          M_pixel=None):

        """

        Computes the total non_match_loss with an l2_pixel norm term

        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_a
        :type matches_b:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a

        :param M_descriptor: margin for descriptor loss term
        :type M_descriptor: float
        :param M_pixel: margin for pixel loss term
        :type M_pixel: float
        :return: non_match_loss, num_hard_negatives
        :rtype: torch.Variable, int
        """

        if M_descriptor is None:
            M_descriptor = self._config["M_descriptor"]

        if M_pixel is None:
            M_pixel = self._config["M_pixel"]

        PCL = PixelwiseContrastiveLoss

        num_non_matches = non_matches_a.size()[0]

        non_match_descriptor_loss, num_hard_negatives, _, _ = PCL.non_match_descriptor_loss(image_a_pred, image_b_pred,
                                                                                            non_matches_a,
                                                                                            non_matches_b,
                                                                                            M=M_descriptor)

        non_match_pixel_l2_loss, _, _ = self.l2_pixel_loss(matches_b, non_matches_b, M_pixel=M_pixel)

        non_match_loss = (non_match_descriptor_loss * non_match_pixel_l2_loss).sum()

        if self.debug:
            self._debug_data['num_hard_negatives'] = num_hard_negatives
            self._debug_data['fraction_hard_negatives'] = num_hard_negatives * 1.0 / num_non_matches

        return non_match_loss, num_hard_negatives

    def non_match_loss_descriptor_only(self, image_a_pred, image_b_pred, non_matches_a, non_matches_b, M_descriptor=0.5,
                                       invert=False):
        """
        Computes the non-match loss, only using the desciptor norm
        :param image_a_pred:
        :type image_a_pred:
        :param image_b_pred:
        :type image_b_pred:
        :param non_matches_a:
        :type non_matches_a:
        :param non_matches_b:
        :type non_matches_b:
        :param M:
        :type M:
        :return: non_match_loss, num_hard_negatives
        :rtype: torch.Variable, int
        """
        PCL = PixelwiseContrastiveLoss

        if M_descriptor is None:
            M_descriptor = self._config["M_descriptor"]

        non_match_loss_vec, num_hard_negatives, _, _ = PCL.non_match_descriptor_loss(image_a_pred, image_b_pred,
                                                                                     non_matches_a,
                                                                                     non_matches_b, M=M_descriptor,
                                                                                     invert=invert)

        num_non_matches = long(non_match_loss_vec.size()[0])

        non_match_loss = non_match_loss_vec.sum()

        if self._debug:
            self._debug_data['num_hard_negatives'] = num_hard_negatives
            self._debug_data['fraction_hard_negatives'] = num_hard_negatives * 1.0 / num_non_matches

        return non_match_loss, num_hard_negatives

    def l2_pixel_loss(self, matches_b, non_matches_b, M_pixel=None):
        """
        Apply l2 loss in pixel space.

        This weights non-matches more if they are "far away" in pixel space.

        :param matches_b: A torch.LongTensor with shape torch.Shape([num_matches])
        :param non_matches_b: A torch.LongTensor with shape torch.Shape([num_non_matches])
        :return l2 loss per sample: A torch.FloatTensorof with shape torch.Shape([num_matches])
        """

        if M_pixel is None:
            M_pixel = self._config['M_pixel']

        num_non_matches_per_match = len(non_matches_b) / len(matches_b)

        ground_truth_pixels_for_non_matches_b = torch.t(
            matches_b.repeat(num_non_matches_per_match, 1)).contiguous().view(-1, 1)

        ground_truth_u_v_b = self.flattened_pixel_locations_to_u_v(ground_truth_pixels_for_non_matches_b)
        sampled_u_v_b = self.flattened_pixel_locations_to_u_v(non_matches_b.unsqueeze(1))

        # each element is always within [0,1], you have 1 if you are at least M_pixel away in
        # L2 norm in pixel space
        norm_degree = 2
        squared_l2_pixel_loss = 1.0 / M_pixel * torch.clamp(
            (ground_truth_u_v_b - sampled_u_v_b).float().norm(norm_degree, 1), max=M_pixel)

        return squared_l2_pixel_loss, ground_truth_u_v_b, sampled_u_v_b

    def flattened_pixel_locations_to_u_v(self, flat_pixel_locations):
        """
        :param flat_pixel_locations: A torch.LongTensor of shape torch.Shape([n,1]) where each element
         is a flattened pixel index, i.e. some integer between 0 and 307,200 for a 640x480 image

        :type flat_pixel_locations: torch.LongTensor

        :return A torch.LongTensor of shape (n,2) where the first column is the u coordinates of
        the pixel and the second column is the v coordinate

        """
        u_v_pixel_locations = flat_pixel_locations.repeat(1, 2)
        u_v_pixel_locations[:, 0] = u_v_pixel_locations[:, 0] % self.image_width
        u_v_pixel_locations[:, 1] = u_v_pixel_locations[:, 1] / self.image_width
        return u_v_pixel_locations

    def get_l2_pixel_loss_original(self):
        pass

    def get_loss_original(self, image_a_pred, image_b_pred, matches_a,
                          matches_b, non_matches_a, non_matches_b,
                          M_margin=0.5, non_match_loss_weight=1.0):

        # this is pegged to it's implemenation at sha 87abdb63bb5b99d9632f5c4360b5f6f1cf54245f
        """
        Computes the loss function
        DCN = Dense Correspondence Network
        num_images = number of images in this batch
        num_matches = number of matches
        num_non_matches = number of non-matches
        W = image width
        H = image height
        D = descriptor dimension
        match_loss = 1/num_matches \sum_{num_matches} ||descriptor_a - descriptor_b||_2^2
        non_match_loss = 1/num_non_matches \sum_{num_non_matches} max(0, M_margin - ||descriptor_a - descriptor_b||_2^2 )
        loss = match_loss + non_match_loss
        :param image_a_pred: Output of DCN network on image A.
        :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
        :param image_b_pred: same as image_a_pred
        :type image_b_pred:
        :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
        :type matches_a: torch.Variable(torch.FloatTensor)
        :param matches_b: same as matches_b
        :type matches_b:
        :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
        to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
        :type non_matches_a: torch.Variable(torch.FloatTensor)
        :param non_matches_b: same as non_matches_a
        :type non_matches_b:
        :return: loss, match_loss, non_match_loss
        :rtype: torch.Variable(torch.FloatTensor) each of shape torch.Size([1])
        """

        num_matches = matches_a.size()[0]
        num_non_matches = non_matches_a.size()[0]

        matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
        matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)

        match_loss = 1.0 / num_matches * (matches_a_descriptors - matches_b_descriptors).pow(2).sum()

        # add loss via non_matches
        non_matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a)
        non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b)
        pixel_wise_loss = (non_matches_a_descriptors - non_matches_b_descriptors).pow(2).sum(dim=2)
        pixel_wise_loss = torch.add(torch.neg(pixel_wise_loss), M_margin)
        zeros_vec = torch.zeros_like(pixel_wise_loss)
        non_match_loss = non_match_loss_weight * 1.0 / num_non_matches * torch.max(zeros_vec, pixel_wise_loss).sum()

        loss = match_loss + non_match_loss

        return loss, match_loss, non_match_loss

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

def create_non_correspondences(uv_b_matches, img_b_shape, num_non_matches_per_match=100, img_b_mask=None):
    """
    Takes in pixel matches (uv_b_matches) that correspond to matches in another image, and generates non-matches by just sampling in image space.

    Optionally, the non-matches can be sampled from a mask for image b.

    Returns non-matches as pixel positions in image b.

    Please see 'coordinate_conventions.md' documentation for an explanation of pixel coordinate conventions.

    ## Note that arg uv_b_matches are the outputs of batch_find_pixel_correspondences()

    :param uv_b_matches: tuple of torch.FloatTensors, where each FloatTensor is length n, i.e.:
        (torch.FloatTensor, torch.FloatTensor)

    :param img_b_shape: tuple of (H,W) which is the shape of the image

    (optional)
    :param num_non_matches_per_match: int

    (optional)
    :param img_b_mask: torch.FloatTensor (can be cuda or not)
        - masked image, we will select from the non-zero entries
        - shape is H x W

    :return: tuple of torch.FloatTensors, i.e. (torch.FloatTensor, torch.FloatTensor).
        - The first element of the tuple is all "u" pixel positions, and the right element of the tuple is all "v" positions
        - Each torch.FloatTensor is of shape torch.Shape([num_matches, non_matches_per_match])
        - This shape makes it so that each row of the non-matches corresponds to the row for the match in uv_a
    """
    image_width = img_b_shape[1]
    image_height = img_b_shape[0]
    # print("uv_b_matches: ", uv_b_matches)
    if uv_b_matches == None:
        return None

    num_matches = len(uv_b_matches[0])

    def get_random_uv_b_non_matches():
        return pytorch_rand_select_pixel(width=image_width, height=image_height,
                                         num_samples=num_matches * num_non_matches_per_match)

    if img_b_mask is not None:
        img_b_mask_flat = img_b_mask.view(-1, 1).squeeze(1)
        mask_b_indices_flat = torch.nonzero(img_b_mask_flat)
        if len(mask_b_indices_flat) == 0:
            print("warning, empty mask b")
            uv_b_non_matches = get_random_uv_b_non_matches()
        else:
            num_samples = num_matches * num_non_matches_per_match
            rand_numbers_b = torch.rand(num_samples) * len(mask_b_indices_flat)
            rand_indices_b = torch.floor(rand_numbers_b).long()
            randomized_mask_b_indices_flat = torch.index_select(mask_b_indices_flat, 0, rand_indices_b).squeeze(1)
            uv_b_non_matches = (
            randomized_mask_b_indices_flat % image_width, randomized_mask_b_indices_flat / image_width)
    else:
        uv_b_non_matches = get_random_uv_b_non_matches()

    # for each in uv_a, we want non-matches
    # first just randomly sample "non_matches"
    # we will later move random samples that were too close to being matches
    uv_b_non_matches = (uv_b_non_matches[0].view(num_matches, num_non_matches_per_match),
                        uv_b_non_matches[1].view(num_matches, num_non_matches_per_match))

    # uv_b_matches can now be used to make sure no "non_matches" are too close
    # to preserve tensor size, rather than pruning, we can perturb these in pixel space
    copied_uv_b_matches_0 = torch.t(uv_b_matches[0].repeat(num_non_matches_per_match, 1))
    copied_uv_b_matches_1 = torch.t(uv_b_matches[1].repeat(num_non_matches_per_match, 1))

    diffs_0 = copied_uv_b_matches_0 - uv_b_non_matches[0].type(dtype_float)
    diffs_1 = copied_uv_b_matches_1 - uv_b_non_matches[1].type(dtype_float)

    diffs_0_flattened = diffs_0.contiguous().view(-1, 1)
    diffs_1_flattened = diffs_1.contiguous().view(-1, 1)

    diffs_0_flattened = torch.abs(diffs_0_flattened).squeeze(1)
    diffs_1_flattened = torch.abs(diffs_1_flattened).squeeze(1)

    need_to_be_perturbed = torch.zeros_like(diffs_0_flattened)
    ones = torch.zeros_like(diffs_0_flattened)
    num_pixels_too_close = 1.0
    threshold = torch.ones_like(diffs_0_flattened) * num_pixels_too_close

    # determine which pixels are too close to being matches
    need_to_be_perturbed = where(diffs_0_flattened < threshold, ones, need_to_be_perturbed)
    need_to_be_perturbed = where(diffs_1_flattened < threshold, ones, need_to_be_perturbed)

    minimal_perturb = num_pixels_too_close / 2
    minimal_perturb_vector = (torch.rand(len(need_to_be_perturbed)) * 2).floor() * (
                minimal_perturb * 2) - minimal_perturb
    std_dev = 10
    random_vector = torch.randn(len(need_to_be_perturbed)) * std_dev + minimal_perturb_vector
    perturb_vector = need_to_be_perturbed * random_vector

    uv_b_non_matches_0_flat = uv_b_non_matches[0].view(-1, 1).type(dtype_float).squeeze(1)
    uv_b_non_matches_1_flat = uv_b_non_matches[1].view(-1, 1).type(dtype_float).squeeze(1)

    uv_b_non_matches_0_flat = uv_b_non_matches_0_flat + perturb_vector
    uv_b_non_matches_1_flat = uv_b_non_matches_1_flat + perturb_vector

    # now just need to wrap around any that went out of bounds

    # handle wrapping in width
    lower_bound = 0.0
    upper_bound = image_width * 1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * upper_bound

    uv_b_non_matches_0_flat = where(uv_b_non_matches_0_flat > upper_bound_vec,
                                    uv_b_non_matches_0_flat - upper_bound_vec,
                                    uv_b_non_matches_0_flat)

    uv_b_non_matches_0_flat = where(uv_b_non_matches_0_flat < lower_bound_vec,
                                    uv_b_non_matches_0_flat + upper_bound_vec,
                                    uv_b_non_matches_0_flat)

    # handle wrapping in height
    lower_bound = 0.0
    upper_bound = image_height * 1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * upper_bound

    uv_b_non_matches_1_flat = where(uv_b_non_matches_1_flat > upper_bound_vec,
                                    uv_b_non_matches_1_flat - upper_bound_vec,
                                    uv_b_non_matches_1_flat)

    uv_b_non_matches_1_flat = where(uv_b_non_matches_1_flat < lower_bound_vec,
                                    uv_b_non_matches_1_flat + upper_bound_vec,
                                    uv_b_non_matches_1_flat)

    return (uv_b_non_matches_0_flat.view(num_matches, num_non_matches_per_match),
            uv_b_non_matches_1_flat.view(num_matches, num_non_matches_per_match))

def where(cond, x_1, x_2):
    """
    We follow the torch.where implemented in 0.4.
    See http://pytorch.org/docs/master/torch.html?highlight=where#torch.where

    For more discussion see https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8


    Return a tensor of elements selected from either x_1 or x_2, depending on condition.
    :param cond: cond should be tensor with entries [0,1]
    :type cond:
    :param x_1: torch.Tensor
    :type x_1:
    :param x_2: torch.Tensor
    :type x_2:
    :return:
    :rtype:
    """
    cond = cond.type(dtype_float)
    return (cond * x_1) + ((1-cond) * x_2)

def pytorch_rand_select_pixel(width,height,num_samples=1):
    two_rand_numbers = torch.rand(2,num_samples)
    two_rand_numbers[0,:] = two_rand_numbers[0,:]*width
    two_rand_numbers[1,:] = two_rand_numbers[1,:]*height
    two_rand_ints    = torch.floor(two_rand_numbers).type(dtype_long)
    return (two_rand_ints[0], two_rand_ints[1])