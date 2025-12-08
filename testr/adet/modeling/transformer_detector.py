import torch
from torch import nn

from detectron2.structures import ImageList, Instances
from testr.adet.modeling.testr.losses import SetCriterion
from testr.adet.modeling.testr.matcher import build_matcher
from testr.adet.modeling.testr.models import TESTR
from testr.adet.utils.misc import NestedTensor, box_xyxy_to_cxcywh


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):

    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])

    # scale bezier points
    if results.has("beziers"):
        beziers = results.beziers
        # scale and clip in place
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y

    if results.has("polygons"):
        polygons = results.polygons
        polygons[:, 0::2] *= scale_x
        polygons[:, 1::2] *= scale_y

    return results


class TransformerDetector(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        self.test_score_threshold = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON
        self.testr = TESTR(cfg)

        box_matcher, point_matcher = build_matcher(cfg)
        
        loss_cfg = cfg.MODEL.TRANSFORMER.LOSS
        weight_dict = {'loss_ce': loss_cfg.POINT_CLASS_WEIGHT, 'loss_ctrl_points': loss_cfg.POINT_COORD_WEIGHT, 'loss_texts': loss_cfg.POINT_TEXT_WEIGHT}
        enc_weight_dict = {'loss_bbox': loss_cfg.BOX_COORD_WEIGHT, 'loss_giou': loss_cfg.BOX_GIOU_WEIGHT, 'loss_ce': loss_cfg.BOX_CLASS_WEIGHT}
        if loss_cfg.AUX_LOSS:
            aux_weight_dict = {}
            # decoder aux loss
            for i in range(cfg.MODEL.TRANSFORMER.DEC_LAYERS - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()})
            # encoder aux loss
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in enc_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        enc_losses = ['labels', 'boxes']
        dec_losses = ['labels', 'ctrl_points', 'texts']

        self.criterion = SetCriterion(self.testr.num_classes, box_matcher, point_matcher,
                                      weight_dict, enc_losses, dec_losses, self.testr.num_ctrl_points, 
                                      focal_alpha=loss_cfg.FOCAL_ALPHA, focal_gamma=loss_cfg.FOCAL_GAMMA)

        self.to(self.device)


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


    def forward(self, extracted_feats, targets, MODE='VAL'):
        # breakpoint()
        output = self.testr(extracted_feats)
        bs = output['pred_logits'].shape[0]
        image_sizes = [(512,512) for _ in range(bs)]
        
        if MODE == 'TRAIN':
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
        elif MODE == 'VAL':
            loss_dict=None
        
        ctrl_point_cls = output["pred_logits"]          
        ctrl_point_coord = output["pred_ctrl_points"]   
        text_pred = output["pred_texts"]                
        results = self.inference(ctrl_point_cls, ctrl_point_coord, text_pred, image_sizes)
        return loss_dict, results
    

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            raw_ctrl_points = targets_per_image.polygons if self.use_polygon else targets_per_image.beziers
            gt_ctrl_points = raw_ctrl_points.reshape(-1, self.testr.num_ctrl_points, 2) / torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            gt_text = targets_per_image.text
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "ctrl_points": gt_ctrl_points, "texts": gt_text})
        return new_targets


    def inference(self, ctrl_point_cls, ctrl_point_coord, text_pred, image_sizes):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []

        text_pred = torch.softmax(text_pred, dim=-1)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        for scores_per_image, labels_per_image, ctrl_point_per_image, text_per_image, image_size in zip(
            scores, labels, ctrl_point_coord, text_pred, image_sizes
        ):
            selector = scores_per_image >= self.test_score_threshold
            scores_per_image = scores_per_image[selector]
            labels_per_image = labels_per_image[selector]
            ctrl_point_per_image = ctrl_point_per_image[selector]
            text_per_image = text_per_image[selector]
            result = Instances(image_size)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            result.rec_scores = text_per_image
            ctrl_point_per_image[..., 0] *= image_size[1]
            ctrl_point_per_image[..., 1] *= image_size[0]
            if self.use_polygon:
                result.polygons = ctrl_point_per_image.flatten(1)
            else:
                result.beziers = ctrl_point_per_image.flatten(1)
            _, topi = text_per_image.topk(1)
            result.recs = topi.squeeze(-1)
            results.append(result)
        return results
