# Copyright (c) OpenMMLab. All rights reserved.
#from cv2 import resize
from mmseg.ops import resize
import mmcv
from mmdet.models import seg_heads
import numpy as np
from numpy.core.fromnumeric import size
import torch
from torch import nn as nn
from torch.nn import functional as F
import warnings
from mmcv.parallel import DataContainer as DC
from mmcv.image import tensor2imgs
from os import path as osp

from mmdet3d.core import (CameraInstance3DBoxes, bbox3d2result,
                          show_multi_modality_result)
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
#from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.detectors.base import BaseDetector
from mmdet.core.visualization import imshow_det_bboxes

from mmseg.models.builder import build_head as seg_build_head
#from ..builder import DETECTORS, build_backbone, build_head, build_neck

from mmdet3d.datasets.poseidon_2d_dataset import Poseidon2DDataset


@DETECTORS.register_module()
class Autopilot2D(BaseDetector):
    """Base class for monocular 3D single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_2d_head=None,
                 seg_decode_head=None,
                 bbox_3d_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(Autopilot2D, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_2d_head.update(train_cfg=train_cfg)
        bbox_2d_head.update(test_cfg=test_cfg)
        self.bbox_2d_head = build_head(bbox_2d_head)
        self.pred_bbox2d = True

        if seg_decode_head is not None:
            self.seg_decode_head = seg_build_head(seg_decode_head)
            self.seg_num_classes = self.seg_decode_head.num_classes
        else:
            self.seg_decode_head = None

        if bbox_3d_head is not None:
            bbox_3d_head.update(train_cfg=train_cfg)
            bbox_3d_head.update(test_cfg=test_cfg)
            self.bbox_3d_head = build_head(bbox_3d_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feats(self, imgs):
        """Directly extract features from the backbone+neck."""
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_2d_head(x)
        if self.seg_decode_head is not None:
            seg = self.seg_decode_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      #gt_bboxes_3d,
                      #gt_labels_3d,
                      #centers2d,
                      #depths,
                      gt_semantic_seg=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_3d (list[Tensor]): Each item are the 3D truth boxes for
                each image in [x, y, z, w, l, h, theta, vx, vy] format.
            gt_labels_3d (list[Tensor]): 3D class indices corresponding to
                each box.
            centers2d (list[Tensor]): Projected 3D centers onto 2D images.
            depths (list[Tensor]): Depth of projected centers on 2D images.
            attr_labels (list[Tensor], optional): Attribute indices
                corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        #losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                      gt_labels, gt_bboxes_3d,
        #                                      gt_labels_3d, centers2d, depths,
        #                                      attr_labels, gt_bboxes_ignore)
        losses = self.bbox_2d_head.forward_train(x, img_metas, gt_bboxes, gt_labels,
                                              attr_labels, gt_bboxes_ignore)
        if self.seg_decode_head is not None:
            seg_losses = self.seg_decode_head.forward_train(x, img_metas,
                                                gt_semantic_seg, self.train_cfg)
            losses.update(seg_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_2d_head(x)
        bbox_outputs = self.bbox_2d_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        if self.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.bbox_2d_head.num_classes)
                for bboxes2d, labels in bbox_outputs
            ]
            bbox_outputs = [bbox_outputs[0][:-1]]
        
        size = img_metas[0]['ori_shape'][:2]
        if self.seg_decode_head is not None:
            seg_logit = self.seg_decode_head(x)
        #    seg_logit = self.seg_decode_head.forward_test(x, img_metas, self.test_cfg)
            seg_logit = resize(
                input=seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.seg_decode_head.align_corners
            )
            seg_pred = seg_logit.argmax(dim=1)

        """
        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.pred_bbox2d:
            for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list
        """
        #if self.seg_decode_head is not None:
        #    return (bbox2d_img, seg_pred)
        #else:
        #    return bbox2d_img
        return bbox2d_img, seg_pred

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        feats = self.extract_feats(imgs)

        # only support aug_test for one sample
        outs_list = [self.bbox_head(x) for x in feats]
        for i, img_meta in enumerate(img_metas):
            if img_meta[0]['pcd_horizontal_flip']:
                for j in range(len(outs_list[i])):  # for each prediction
                    if outs_list[i][j][0] is None:
                        continue
                    for k in range(len(outs_list[i][j])):
                        # every stride of featmap
                        outs_list[i][j][k] = torch.flip(
                            outs_list[i][j][k], dims=[3])
                reg = outs_list[i][1]
                for reg_feat in reg:
                    # offset_x
                    reg_feat[:, 0, :, :] = 1 - reg_feat[:, 0, :, :]
                    # velo_x
                    if self.bbox_head.pred_velo:
                        reg_feat[:, 7, :, :] = -reg_feat[:, 7, :, :]
                    # rotation
                    reg_feat[:, 6, :, :] = -reg_feat[:, 6, :, :] + np.pi

        merged_outs = []
        for i in range(len(outs_list[0])):  # for each prediction
            merged_feats = []
            for j in range(len(outs_list[0][i])):
                if outs_list[0][i][0] is None:
                    merged_feats.append(None)
                    continue
                # for each stride of featmap
                avg_feats = torch.mean(
                    torch.cat([x[i][j] for x in outs_list]),
                    dim=0,
                    keepdim=True)
                if i == 1:  # regression predictions
                    # rot/velo/2d det keeps the original
                    avg_feats[:, 6:, :, :] = \
                        outs_list[0][i][j][:, 6:, :, :]
                if i == 2:
                    # dir_cls keeps the original
                    avg_feats = outs_list[0][i][j]
                merged_feats.append(avg_feats)
            merged_outs.append(merged_feats)
        merged_outs = tuple(merged_outs)

        bbox_outputs = self.bbox_head.get_bboxes(
            *merged_outs, img_metas[0], rescale=rescale)
        if self.bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.bbox_head.num_classes)
                for bboxes, scores, labels, attrs, bboxes2d in bbox_outputs
            ]
            bbox_outputs = [bbox_outputs[0][:-1]]

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = dict()
        bbox_list.update(img_bbox=bbox_img[0])
        if self.bbox_head.pred_bbox2d:
            bbox_list.update(img_bbox2d=bbox2d_img[0])

        return [bbox_list]

    def show_det_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result[0], None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def show_seg_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        #assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img


    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (list[dict]): Input images and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        dets, segs = result
        segs = segs.cpu().numpy()
        segs = list(segs)

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            if out_dir:
                det_out_file = osp.join(out_dir, 'dets', img_meta['ori_filename'])
                seg_out_file = osp.join(out_dir, 'segs', img_meta['ori_filename'])
            else:
                det_out_file = None
                seg_out_file = None

            self.show_seg_result(
                img_show,
                segs,
                palette=self.PALETTE,
                show=True,
                out_file=seg_out_file,
                opacity=0.5)

            self.show_det_result(
                img_show,
                dets,
                show=True,
                out_file=det_out_file,
                score_thr=0.3)

        #for batch_id in range(len(result)):
        #    if isinstance(data['img_metas'][0], DC):
        #        img_filename = data['img_metas'][0]._data[0][batch_id][
        #            'filename']
        #        cam2img = data['img_metas'][0]._data[0][batch_id]['cam2img']
        #    elif mmcv.is_list_of(data['img_metas'][0], dict):
        #        img_filename = data['img_metas'][0][batch_id]['filename']
        #        cam2img = data['img_metas'][0][batch_id]['cam2img']
        #    else:
        #        ValueError(
        #            f"Unsupported data type {type(data['img_metas'][0])} "
        #            f'for visualization!')
        #    img = mmcv.imread(img_filename)
        #    file_name = osp.split(img_filename)[-1].split('.')[0]

        #    assert out_dir is not None, 'Expect out_dir, got none.'

        #    pred_bboxes = result[batch_id]['img_bbox']['boxes_3d']
        #    assert isinstance(pred_bboxes, CameraInstance3DBoxes), \
        #        f'unsupported predicted bbox type {type(pred_bboxes)}'

        #    show_multi_modality_result(
        #        img,
        #        None,
        #        pred_bboxes,
        #        cam2img,
        #        out_dir,
        #        file_name,
        #        'camera',
        #        show=True)

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels
