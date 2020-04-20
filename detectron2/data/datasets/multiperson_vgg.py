# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import os,json,cv2
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

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
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        

        filename = os.path.join(img_dir, v["filename"])
        filename=filename.split('.')[0]+'_depth'+'.png'
        if not os.path.isfile(filename):
            print('{} not exist!'.format(filename))
            continue
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
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
    return dataset_dicts


def register_multiperson_vgg():
    DatasetCatalog.register('tracker_train', lambda: load_vgg_instances('datasets/tracker_train'))
    MetadataCatalog.get('tracker_train').set(
        thing_classes=CLASS_NAMES)
    DatasetCatalog.register('tracker_val', lambda: load_vgg_instances('datasets/tracker_val'))
    MetadataCatalog.get('tracker_val').set(
        thing_classes=CLASS_NAMES)
        