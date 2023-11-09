
import os
import math
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
def initPatch(initPosition, mapCorner, mapsize, folder, patchsize, step):
    lon, lat = initPosition['lon'], initPosition['lat']
    left_lon, right_lon, bottom_lat, top_lat = mapCorner['left_lon'], mapCorner['right_lon'], mapCorner['bottom_lat'], mapCorner['top_lat']
    map_height, map_width = mapsize['height'], mapsize['width']
    patch_h, patch_w = patchsize['height'], patchsize['width']



    lat_diff = top_lat - lat
    lon_diff = lon - left_lon

    # row_index = int(lat_diff * map_height / (patch_height * (top_lat - bottom_lat)))
    # col_index = int(lon_diff * map_width / (patch_width * (right_lon - left_lon)))
    # initPatch_name = str(row_index * patch_height) + '_' + str(col_index * patch_width) + ".jpg"

    a = (lat_diff * map_height /  (top_lat - bottom_lat))
    b = (lon_diff * map_width /  (right_lon - left_lon))
    # row_index = math.floor((lat_diff * map_height / (step * (top_lat - bottom_lat))))
    # col_index = math.floor((lon_diff * map_width / (step * (right_lon - left_lon))))


    row_index = math.floor( max(0, a - patch_h) / step ) + 1
    col_index = math.floor( max(0 ,b - patch_w) / step ) + 1

    initPatch_name = str(row_index * step) + '_' + str(col_index * step) + ".jpg"

    initPatch_path = os.path.join(folder, initPatch_name)
    return initPatch_path


def searchSpace(curretPatch, step, khop, folder):
    import re


    numbers = re.findall(r'\d+', curretPatch)
    a, b = int(numbers[0]), int(numbers[1])
    # a, b = int(curretPatch.split("_")[0]), int(curretPatch.split("_")[1].split(".")[0])
    neighboring_patch = []
    for i in range(2 * khop + 1):
        for j in range(2 * khop + 1):
            patch_name =  str(a + (i - khop) * step) + '_' + str(b + (j - khop) * step) + ".jpg"
            patch_path = os.path.join(folder, patch_name)
            neighboring_patch.append(patch_path)

    return neighboring_patch

def clipSpecificPatch(Position, mapCorner, mapsize, map_path, patchsize):
    lon, lat = Position['lon'], Position['lat']
    left_lon, right_lon, bottom_lat, top_lat = mapCorner['left_lon'], mapCorner['right_lon'], mapCorner['bottom_lat'], \
                                               mapCorner['top_lat']
    map_height, map_width = mapsize['height'], mapsize['width']
    patch_height, patch_width = patchsize['height'], patchsize['width']

    lat_diff = top_lat - lat
    lon_diff = lon - left_lon

    row = math.floor((lat_diff * map_height / (top_lat - bottom_lat)))
    col = math.floor((lon_diff * map_width / (right_lon - left_lon)))

    img = cv2.imread(map_path, 0)
    cropImg = img[int(row-patch_height/2):int(row+patch_height/2), int(col-patch_width/2):int(col+patch_width/2)]

    # cv2.drawMarker(cropImg,  (int(patch_width/2), int(patch_height/2)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
    #                markerSize=300, thickness=3, line_type=cv2.LINE_AA)
    # plt.figure()
    # plt.imshow(cropImg)
    # plt.show()

    return cropImg, row-patch_height/2, col-patch_width/2


