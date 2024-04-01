r""" Visual Prompt Encoder of VRP-SAM """
from functools import reduce
from operator import add
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.base.resnet as models
import model.base.vgg as vgg_models
from torch.nn import BatchNorm2d as BatchNorm
from common.utils import get_stroke_preset, get_random_points_from_mask, get_mask_by_input_strokes

from .base.transformer_decoder import transformer_decoder

# copy from SEEM
def get_bounding_boxes(mask):
        """
        Returns:
            Boxes: tight bounding boxes around bitmasks.
            If a mask is empty, it's bounding box will be all zero.
        """
        boxes = torch.zeros(mask.shape[0], 4, dtype=torch.float32).to(mask.device)
        box_mask = torch.zeros_like(mask).to(mask.device)
        x_any = torch.any(mask, dim=1)
        y_any = torch.any(mask, dim=2)
        for idx in range(mask.shape[0]):
            x = torch.where(x_any[idx, :])[0].int()
            y = torch.where(y_any[idx, :])[0].int()
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = torch.as_tensor(
                    [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
                )
                x1, y1, x2, y2 = x[0], y[0], x[-1] + 1, y[-1] + 1
        
                box_mask[idx, y1:y2, x1:x2]=1
        return boxes, box_mask

def get_point_mask(mask, training, max_points=20):
        """
        Returns:
            Point_mask: random 20 point for train and test.
            If a mask is empty, it's Point_mask will be all zero.
        """
        max_points = min(max_points, mask.sum().item())
        if training:
            num_points = random.Random().randint(1, max_points) # get a random number of points 
        else:
            num_points = max_points 
        b,h,w = mask.shape
        point_masks = []

        for idx in range(b):
            view_mask = mask[idx].view(-1)
            non_zero_idx = view_mask.nonzero()[:,0] # get non-zero index of mask
            selected_idx = torch.randperm(len(non_zero_idx))[:num_points] # select id
            non_zero_idx = non_zero_idx[selected_idx] # select non-zero index
            rand_mask = torch.zeros(view_mask.shape).to(mask.device) # init rand mask
            rand_mask[non_zero_idx] = 1 # get one place to zero
            point_masks.append(rand_mask.reshape(h, w).unsqueeze(0))
        return torch.cat(point_masks, 0)

def get_scribble_mask(mask, training, stroke_preset=['rand_curve', 'rand_curve_small'], stroke_prob=[0.5, 0.5]):
        """
        Returns:
            Scribble_mask: random 20 point for train and test.
            If a mask is empty, it's Scribble_mask will be all zero.
        """
        if training:
            stroke_preset_name = random.Random().choices(stroke_preset, weights=stroke_prob, k=1)[0]
            nStroke = random.Random().randint(1, min(20, mask.sum().item()))
        else:
            stroke_preset_name = random.Random(321).choices(stroke_preset, weights=stroke_prob, k=1)[0]
            nStroke = random.Random(321).randint(1, min(20, mask.sum().item()))
        preset = get_stroke_preset(stroke_preset_name)
        
        b,h,w = mask.shape
        
        scribble_masks = []
        for idx in range(b):
            points = get_random_points_from_mask(mask[idx].bool(), n=nStroke)  
            rand_mask = get_mask_by_input_strokes(init_points=points, imageWidth=w, imageHeight=h, nStroke=min(nStroke, len(points)), **preset)
            rand_mask = (~torch.from_numpy(rand_mask)) * mask[idx].bool().cpu()
            scribble_masks.append(rand_mask.float().unsqueeze(0))
        return torch.cat(scribble_masks, 0).to(mask.device)

def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs, targets = inputs.flatten(1), targets.flatten(1)
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

class VRP_encoder(nn.Module):
    def __init__(self, args, backbone, use_original_imgsize):
        super(VRP_encoder, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=True)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=True)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.layer0.eval(), self.layer1.eval(), self.layer2.eval(), self.layer3.eval(), self.layer4.eval()
        if backbone == 'vgg16':
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512 
        hidden_dim = 256
        self.downsample_query = nn.Sequential(
            nn.Conv2d(fea_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        
        self.merge_1 = nn.Sequential(
            nn.Conv2d(hidden_dim*2+1, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.num_query = args.num_query

        self.transformer_decoder = transformer_decoder(args, args.num_query, hidden_dim, hidden_dim*2)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()

    def forward(self, condition, query_img, support_img, support_mask, training):

        # if training:
        #     condition = random.Random().choices(['scribble', 'point', 'box', 'mask'], weights=[0.25,0.25,0.25,0.25], k=1)[0]  
        # else:
        #     condition = condition

        if condition == 'scribble':
            support_mask_ori = get_scribble_mask(support_mask,training) # scribble_mask
        elif condition == 'point':
            support_mask_ori = get_point_mask(support_mask,training) # point_mask
        elif condition == 'box':
            boxes, support_mask_ori = get_bounding_boxes(support_mask) # box_mask
        elif condition == 'mask':
            support_mask_ori = support_mask

        with torch.no_grad():
            query_feat_0 = self.layer0(query_img)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.backbone_type == 'vgg16':
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)
            query_feat = torch.cat([query_feat_2, query_feat_3], 1) 

            supp_feat_0 = self.layer0(support_img)
            supp_feat_1 = self.layer1(supp_feat_0)
            supp_feat_2 = self.layer2(supp_feat_1)
            supp_feat_3 = self.layer3(supp_feat_2)
            support_mask = F.interpolate(support_mask_ori.unsqueeze(1).float(), size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='nearest')
            supp_feat_4 = self.layer4(supp_feat_3*support_mask)
            if self.backbone_type == 'vgg16':
                supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)
            supp_feat = torch.cat([supp_feat_2, supp_feat_3], 1)
            pseudo_mask = self.get_pseudo_mask(supp_feat_4, query_feat_4, support_mask)
        
        query_feat = self.downsample_query(query_feat)
        supp_feat = self.downsample_query(supp_feat)
        prototype = self.mask_feature(supp_feat, support_mask)
        supp_feat_bin = prototype.repeat(1, 1, query_feat.shape[2], query_feat.shape[3])
        supp_feat_1 = self.merge_1(torch.cat([supp_feat, supp_feat_bin, support_mask*10], 1))                                                                                    
        query_feat_1 = self.merge_1(torch.cat([query_feat, supp_feat_bin, pseudo_mask*10], 1))

        protos = self.transformer_decoder(query_feat_1, supp_feat_1, support_mask)
        return protos, support_mask_ori

    def mask_feature(self, features, support_mask):
        mask = support_mask
        supp_feat = features * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        return supp_feat

    def predict_mask_nshot(self, args, batch, sam_model, nshot, input_point=None):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        protos_set = []
        for s_idx in range(nshot):
            protos_sub, support_mask = self(args.condition, batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx], False)
            protos_set.append(protos_sub)
        if nshot > 1:
            protos = torch.cat(protos_set, dim=1)
        else:
            protos = protos_sub

        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos,input_point)
        logit_mask = low_masks
        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        pred_mask = torch.sigmoid(logit_mask) >= 0.5

        pred_mask = pred_mask.float()
            
        logit_mask_agg += pred_mask.squeeze(1).clone()
        return logit_mask_agg, support_mask, logit_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        loss_bce = self.bce_with_logits_loss(logit_mask.squeeze(1), gt_mask.float())
        loss_dice = dice_loss(logit_mask, gt_mask, bsz)
        return loss_bce + loss_dice
        

    def train_mode(self):
        self.train()
        self.apply(fix_bn)
        self.layer0.eval(), self.layer1.eval(), self.layer2.eval(), self.layer3.eval(), self.layer4.eval()

    def get_pseudo_mask(self, tmp_supp_feat, query_feat_4, mask):
        resize_size = tmp_supp_feat.size(2)
        tmp_mask = F.interpolate(mask, size=(resize_size, resize_size), mode='bilinear', align_corners=True)

        tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
        q = query_feat_4
        s = tmp_supp_feat_4
        bsize, ch_sz, sp_sz, _ = q.size()[:]

        tmp_query = q
        tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s               
        tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1) 
        tmp_supp = tmp_supp.permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

        cosine_eps = 1e-7
        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
        similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)
        corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
        return corr_query