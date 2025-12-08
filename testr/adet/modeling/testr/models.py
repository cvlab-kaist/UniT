import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from testr.adet.layers.deformable_transformer import DeformableTransformer

from testr.adet.layers.pos_encoding import PositionalEncoding1D
from testr.adet.utils.misc import NestedTensor, inverse_sigmoid_offset, nested_tensor_from_tensor_list, sigmoid_offset
from testr.adet.layers.pos_encoding import PositionalEncoding2D

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FeatFusionBlock(nn.Module):
    def __init__(self, in_ch=4608, d_model=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, 1024, 1),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
            nn.Conv2d(1024, d_model, 1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GroupNorm(32, d_model),
            nn.GELU(),
        )
        self.shortcut = nn.Conv2d(in_ch, d_model, 1)  # residual

    def forward(self, x):
        return self.proj(x) + self.shortcut(x)
    
    
class TESTR(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.d_model                 = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead                   = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers      = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers      = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward         = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout                 = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation              = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels      = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points            = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points            = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.num_proposals           = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.pos_embed_scale         = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_ctrl_points         = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        self.num_classes             = 1
        self.max_text_len            = cfg.MODEL.TRANSFORMER.NUM_CHARS
        self.voc_size                = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.sigmoid_offset          = not cfg.MODEL.TRANSFORMER.USE_POLYGON

        self.text_pos_embed   = PositionalEncoding1D(self.d_model, normalize=True, scale=self.pos_embed_scale)
        
        self.transformer = DeformableTransformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout, activation=self.activation, return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels, dec_n_points=self.dec_n_points, 
            enc_n_points=self.enc_n_points, num_proposals=self.num_proposals,
        )
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)
        self.text_class = nn.Linear(self.d_model, self.voc_size + 1)

        # shared prior between instances (objects)
        self.ctrl_point_embed = nn.Embedding(self.num_ctrl_points, self.d_model)
        self.text_embed = nn.Embedding(self.max_text_len, self.d_model)


        self.diff_feat_proj = nn.ModuleList([
            FeatFusionBlock(in_ch=384*2*6, d_model=self.d_model)  # one block per feature level
            for _ in range(4)
        ])
        
        for proj in self.diff_feat_proj:
            nn.init.xavier_uniform_(proj.proj[0].weight, gain=1)
            nn.init.constant_(proj.proj[0].bias, 0)



        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)

        num_pred = self.num_decoder_layers
        self.ctrl_point_class = nn.ModuleList(
            [self.ctrl_point_class for _ in range(num_pred)])
        self.ctrl_point_coord = nn.ModuleList(
            [self.ctrl_point_coord for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None

        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord

        self.to(self.device)
        self.pos_enc_2d= PositionalEncoding2D(128, normalize=True)


    def forward(self, samples):
        
        extracted_feats = samples
        pos = [self.pos_enc_2d(x) for x in extracted_feats]
        
        srcs = []
        masks = []

        num_feats = len(extracted_feats)
        num_idx_range = num_feats//4

        for l in range(4):
            start_idx = num_idx_range * l
            end_idx = num_idx_range * (l + 1)

            feats = torch.concat(extracted_feats[start_idx:end_idx], dim=1)
            b, _, feat_H, feat_W = feats.shape

            # Use FeatFusionBlock
            fused_feats = self.diff_feat_proj[l](feats)
            srcs.append(fused_feats)

            masks.append(torch.zeros(b, feat_H, feat_W, dtype=torch.bool, device=feats.device))


        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)                     
        text_pos_embed = self.text_pos_embed(self.text_embed.weight)[None, ...].repeat(self.num_proposals, 1, 1)        
        text_embed = self.text_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)                                 

        hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, enc_box_ref_point = self.transformer(
            srcs, masks, pos, ctrl_point_embed, text_embed, text_pos_embed, text_mask=None)

        outputs_classes = []
        outputs_coords = []
        outputs_texts = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)
            outputs_class = self.ctrl_point_class[lvl](hs[lvl])
            tmp = self.ctrl_point_coord[lvl](hs[lvl])
            if reference.shape[-1] == 2:
                tmp += reference[:, :, None, :]
            else:
                assert reference.shape[-1] == 4
                tmp += reference[:, :, None, :2]
            outputs_texts.append(self.text_class(hs_text[lvl]))
            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # breakpoint()
        outputs_class = torch.stack(outputs_classes)    
        outputs_coord = torch.stack(outputs_coords)     
        outputs_text = torch.stack(outputs_texts)       

        out = {'pred_logits': outputs_class[-1],        
               'pred_ctrl_points': outputs_coord[-1],   
               'pred_texts': outputs_text[-1]}          
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_text)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord, 'pred_filtered_boxes': enc_box_ref_point}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_text):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_ctrl_points': b, 'pred_texts': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_text[:-1])]