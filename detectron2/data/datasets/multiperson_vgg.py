# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import os,json,cv2
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from tqdm import tqdm

__all__ = ["register_multiperson_vgg"]


# fmt: off
CLASS_NAMES = [
    "person"
]
# fmt: on


def load_vgg_instances(img_dir):
    """
    Load VGG detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    json_file = os.path.join(img_dir, "annotation.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    total_bad_igms=0
    f_bad = open("bad_imgs.txt", "w")
    print('Building Dataset, please wait')
    for idx, v in enumerate(tqdm(imgs_anns.values())):
        record = {}
        img=None
        filename = os.path.join(img_dir, v["filename"])
        filename_depth=filename.split('.')[0]+'_depth'+'.png'
        if os.path.isfile(filename):
            img=filename
        elif os.path.isfile(filename_depth):
            img=filename_depth
        else:
            f_bad.write(filename_depth+'\n')
            print('file not exist {}'.format(filename_depth))
            total_bad_igms+=1
            continue
        tmp = cv2.imread(img)
        if tmp is None:
            print('img {} corrupted'.format(img))
            total_bad_igms+=1
            continue
        height, width=tmp.shape[:2]
        record["file_name"] = img
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        validate=True
        for _, anno in annos.items():
            if anno["region_attributes"]:
                validate=False
                break
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            if len(px)!=len(py) or len(px)<3 or len(py)<3:
                print('warning! wrong annotation ')
                validate=False
                break
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        if validate:
            record["annotations"] = objs
            dataset_dicts.append(record)
    print('Total bad images {}'.format(total_bad_igms))
    f_bad.close()
    return dataset_dicts


def register_multiperson_vgg():
    DatasetCatalog.register('tracker_train', lambda: load_vgg_instances('datasets/tracker_train/depth'))
    MetadataCatalog.get('tracker_train').set(
        thing_classes=CLASS_NAMES)
    DatasetCatalog.register('tracker_val', lambda: load_vgg_instances('datasets/tracker_val/depth'))
    MetadataCatalog.get('tracker_val').set(
        thing_classes=CLASS_NAMES)
    DatasetCatalog.register('mh_train', lambda: load_vgg_instances('datasets/mh_train'))
    MetadataCatalog.get('mh_train').set(
        thing_classes=CLASS_NAMES)
    DatasetCatalog.register('mh_val', lambda: load_vgg_instances('datasets/mh_val'))
    MetadataCatalog.get('mh_val').set(
        thing_classes=CLASS_NAMES)        
