import os
import copy
import mmcv
import json
import random
import numpy as np
import pyquaternion
import tempfile
import torch
import warnings
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from mmdet3d.core import bbox3d2result, box3d_multiclass_nms, xywhr2xyxyr
from mmdet.datasets import DATASETS, CocoDataset, CustomDataset
from ..core import show_multi_modality_result
from ..core.bbox import CameraInstance3DBoxes, get_box_type
from .pipelines import Compose
from .utils import extract_result_dict, get_loading_pipeline


@DATASETS.register_module()
class Poseidon2DDataset(CustomDataset):

    CLASSES = ('VehicleFull', 'Pedestrian', 'TrafficSign',
        'TrafficLight', 'TrafficArrow', 'TrafficCone', 'Cyclist',
        'Headlights', 'Taillights', 'Streetlight')

    CAT_DICT = {0:'VehicleRear', 1:'VehicleFull', 2:'Pedestrian', 5:'TrafficSign',
        6:'TrafficLight', 11:'TrafficArrow', 13:'TrafficCone', 16:'Cyclist',
        26:'Headlights', 27:'Taillights', 28:'Streetlight'}

    SEG_CLASSES = (
        'road', 'background', 'fance', 'pole', 'traffic sign', 'person', 'vehicle',
        'bicycle', 'lane marking', 'crosswalk', 'traffic arrow', 'speed mark',
        'guide line', 'traffic cone', 'stop line', 'speed bumps'
        )

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
           [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
           [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
           [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
           ]

    IMAGE_WIDTH = 3840
    IMAGE_HEIGHT = 2160

    def __init__(self, min_size=None, **kwargs):
        assert self.CLASSES or kwargs.get(
            'classes', None), 'CLASSES in `XMLDataset` can not be None.'
        super(Poseidon2DDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def load_annotations(self, ann_file, img_suffix='.jpg', seg_suffix='.png'):
        self.tasks = list()
        with open(ann_file, 'r') as f:
            tasks = f.read().splitlines()
            assert type(
               tasks
            ) == list, 'annotation file format {} not supported'.format(
                type(tasks))
            self.tasks = tasks

        data_infos = []
        for task in self.tasks:
            task_dir = os.path.join(os.path.dirname(ann_file), task)
            lpath = osp.join(task_dir, 'list.txt')
            with open(lpath, 'r') as lf:
                ids = lf.read().splitlines()
            #img_names = os.listdir(task_dir + '/images/')
            SEL_NUM = 10
            if len(ids) < SEL_NUM:
                continue
            sel_names = random.sample(ids, SEL_NUM)
            for name in sel_names:
                im_path = os.path.join(task_dir, 'images', name + img_suffix)
                seg_path = os.path.join(task_dir, 'segmentation', name + seg_suffix)
                label_path = os.path.join(task_dir, 'results',
                    'obstacle_2d', name + '.json')
                if not os.path.exists(label_path):
                    continue
                data_infos.append(
                    dict(filename=im_path, label_path=label_path,
                        width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT,
                        seg_map=seg_path)
                )

        return data_infos

    def _filter_imgs(self, min_size=32):
        valid_inds = list(range(len(self.data_infos)))
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        bboxes = []
        labels = []
        seg_map = self.data_infos[idx]['seg_map']
        label_path = self.data_infos[idx]['label_path']
        label_obs2d = json.load(open(label_path))
        for obj in label_obs2d['obstacles']:
            md = obj['model']
            if md not in self.CAT_DICT:
                continue #skip vehicle rear
            elif self.CAT_DICT[md] == 'VehicleRear':
                continue
                #print(md)
            cat = self.CAT_DICT[md]
            label = self.cat2label[cat]
            bbox = [
                int(obj['rect']['left']),
                int(obj['rect']['top']),
                int(obj['rect']['right']),
                int(obj['rect']['bottom'])
            ]
            bboxes.append(bbox)
            labels.append(label)

        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            seg_map=seg_map)
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        label_path = self.data_infos[idx]['label_path']
        label_obs2d = json.load(open(label_path))
        for obj in label_obs2d['obstacles']:
            label = obj['model']
            cat_ids.append(label)
        return cat_ids

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)

        """ for semantic_seg data """
        # load segmentation
        file_client = mmcv.FileClient(backend='disk')
        filename = results['ann_info']['seg_map']
        img_bytes = file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend='pillow').squeeze().astype(np.uint8)
        # avoid using underflow conversion
        gt_semantic_seg[gt_semantic_seg == 1] = 255
        gt_semantic_seg[gt_semantic_seg == 0] = 1
        gt_semantic_seg = gt_semantic_seg - 1
        gt_semantic_seg[gt_semantic_seg == 254] = 255
        gt_semantic_seg = mmcv.impad(
            gt_semantic_seg, shape=(270,480), pad_val=255
        )
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        """ end """

        return self.pipeline(results)