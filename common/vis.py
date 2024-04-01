r""" Visualize model predictions """
import os

from PIL import Image
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import torchvision.transforms as transforms

from . import utils


class Visualizer:

    @classmethod
    def initialize(cls, visualize):
        cls.visualize = visualize
        if not visualize:
            return

        cls.colors = {'red': (255, 50, 50), 'blue': (0, 0, 255), 'green': (0, 255, 0),}
        for key, value in cls.colors.items():
            cls.colors[key] = tuple([c / 255 for c in cls.colors[key]])

        cls.mean_img = [0.485, 0.456, 0.406]
        cls.std_img = [0.229, 0.224, 0.225]
        cls.to_pil = transforms.ToPILImage()
        cls.vis_path = './vis/'
        if not os.path.exists(cls.vis_path): os.makedirs(cls.vis_path)

    @classmethod
    def visualize_prediction_batch(cls, spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b, batch_idx, iou_b=None):
        spt_img_b = utils.to_cpu(spt_img_b)
        spt_mask_b = utils.to_cpu(spt_mask_b)
        qry_img_b = utils.to_cpu(qry_img_b)
        qry_mask_b = utils.to_cpu(qry_mask_b)
        pred_mask_b = utils.to_cpu(pred_mask_b)
        cls_id_b = utils.to_cpu(cls_id_b)

        for sample_idx, (spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id) in \
                enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b)):
            iou = iou_b[sample_idx] if iou_b is not None else None
            cls.visualize_prediction(spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sample_idx, True, iou)

    @classmethod
    def visualize_prediction_batch_demo(cls, spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b, batch_idx, sub_idx, iou_b=None):
        spt_img_b = utils.to_cpu(spt_img_b)
        spt_mask_b = utils.to_cpu(spt_mask_b)
        qry_img_b = utils.to_cpu(qry_img_b)
        qry_mask_b = utils.to_cpu(qry_mask_b)
        pred_mask_b = utils.to_cpu(pred_mask_b)
        cls_id_b = utils.to_cpu(cls_id_b)

        for sample_idx, (spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id) in \
                enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b)):
            iou = iou_b[sample_idx] if iou_b is not None else None
            cls.visualize_prediction_demo(spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sample_idx, sub_idx, True, iou)
    
    @classmethod  
    def visualize_sub_prediction_batch(cls, qry_img_b, batch_idx, pred_mask_b):
        
        qry_img_b = utils.to_cpu(qry_img_b)
        # import pdb; pdb.set_trace()
        pred_mask_b = utils.to_cpu(pred_mask_b)
        # pred_mask_b = pred_mask_b

        for sample_idx, (qry_img, pred_mask) in enumerate(zip(qry_img_b, pred_mask_b)):
            for idx in range(pred_mask.shape[0]):
                # import pdb; pdb.set_trace()
                sub_merge_img = cls.visualize_sub_prediction(qry_img, pred_mask[idx,:,:])
                sub_path = cls.vis_path + '%d_img/'% (batch_idx)
                if not os.path.exists(sub_path): os.makedirs(sub_path)
                sub_merge_img.save(sub_path + '%d_sub-%d' % (sample_idx, idx) + '.jpg')

    @classmethod
    def to_numpy(cls, tensor, type):
        if type == 'img':
            return np.array(cls.to_pil(cls.unnormalize(tensor))).astype(np.uint8)
        elif type == 'mask':
            return np.array(tensor).astype(np.uint8)
        else:
            raise Exception('Undefined tensor type: %s' % type)

    @classmethod
    def visualize_prediction(cls, spt_imgs, spt_masks, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sample_idx, label, iou=None):
        
        spt_color = cls.colors['blue']
        qry_color = cls.colors['red']
        pred_color = cls.colors['red']

        spt_imgs = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        spt_imgs_ori = [Image.fromarray(spt_img) for spt_img in spt_imgs]
        spt_pils = [cls.to_pil(spt_img) for spt_img in spt_imgs]
        spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img, spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs, spt_masks)]
        mask_gt = [Image.fromarray(cls.apply_mask_gt(spt_mask, spt_color)) for spt_mask in spt_masks]
        # import pdb; pdb.set_trace()
        qry_img = cls.to_numpy(qry_img, 'img')
        qry_img_ori = Image.fromarray(qry_img)
        qry_pil = cls.to_pil(qry_img)
        qry_mask = cls.to_numpy(qry_mask, 'mask')
        pred_mask = cls.to_numpy(pred_mask, 'mask')
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        # pred_masked_pil = cls.apply_point(pred_masked_pil, postivate_pos)
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))

        merged_pil = cls.merge_image_pair(mask_gt + spt_masked_pils + [pred_masked_pil, qry_masked_pil])

        iou = iou.item() if iou else 0.0
        merged_pil.save(cls.vis_path + 'class-%d_iou-%.2f_%d' % (cls_id, iou, batch_idx) + '.jpg')

    @classmethod
    def visualize_prediction_demo(cls, spt_imgs, spt_masks, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sub_idx, sample_idx, label, iou=None):
        
        spt_color = cls.colors['blue']
        qry_color = cls.colors['red']
        pred_color = cls.colors['red']

        spt_imgs = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        spt_imgs_ori = [Image.fromarray(spt_img) for spt_img in spt_imgs]
        spt_pils = [cls.to_pil(spt_img) for spt_img in spt_imgs]
        spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img, spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs, spt_masks)]
        mask_gt = [Image.fromarray(cls.apply_mask_gt(spt_mask, spt_color)) for spt_mask in spt_masks]
        # import pdb; pdb.set_trace()
        qry_img = cls.to_numpy(qry_img, 'img')
        qry_img_ori = Image.fromarray(qry_img)
        qry_pil = cls.to_pil(qry_img)
        qry_mask = cls.to_numpy(qry_mask, 'mask')
        pred_mask = cls.to_numpy(pred_mask, 'mask')
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        # pred_masked_pil = cls.apply_point(pred_masked_pil, postivate_pos)
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))

        save_path_1 = os.path.join(cls.vis_path,'%d_refer/'%(batch_idx))
        if not os.path.exists(save_path_1): os.makedirs(save_path_1)
        # import pdb; pdb.set_trace()
        mask_gt[0].save(save_path_1 + ' 0_mask_gt'+'.jpg')
        spt_imgs_ori[0].save(save_path_1 + ' 0_qry_img'+'.jpg')
        spt_masked_pils[0].save(save_path_1 + ' 1_re_mask_img'+'.jpg')
        pred_masked_pil.save(save_path_1 + ' 2_tgt_mask_img_%d'%(sample_idx)+'.jpg')

    @classmethod
    def visualize_sub_prediction(cls, qry_img, pred_mask):
        
        spt_color = cls.colors['blue']
        qry_color = cls.colors['red']
        pred_color = cls.colors['red']

        qry_img = cls.to_numpy(qry_img, 'img')
        qry_pil = cls.to_pil(qry_img)
        # qry_mask = cls.to_numpy(qry_mask, 'mask')
        pred_mask = cls.to_numpy(pred_mask, 'mask')
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        
        return pred_masked_pil


    @classmethod
    def merge_image_pair(cls, pil_imgs):
        r""" Horizontally aligns a pair of pytorch tensor images (3, H, W) and returns PIL object """

        canvas_width = sum([pil.size[0] for pil in pil_imgs])
        canvas_height = max([pil.size[1] for pil in pil_imgs])
        canvas = Image.new('RGB', (canvas_width, canvas_height))

        xpos = 0
        for pil in pil_imgs:
            canvas.paste(pil, (xpos, 0))
            xpos += pil.size[0]

        return canvas

    @classmethod
    def apply_mask(cls, image, mask, color, alpha=0.5):
        r""" Apply mask to the given image. """

        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image
    
    @classmethod
    def apply_mask_gt(cls, mask, color, alpha=1):
        r""" Apply mask to the given image. """
        image = Image.new('RGB', (512,512),(255,255,255))
        # image_copy = image.copy()
        image = np.array(image)  
        # import pdb; pdb.set_trace()     
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    @classmethod
    def apply_point(cls, image, points):
        r""" Apply mask to the given image. """
        draw = ImageDraw.Draw(image)
        if points.shape[0]==0:
            return image
        else:
            for point in points:
                draw.rectangle((point[0]-5, point[1]-5, point[0]+5, point[1]+5), fill=(102, 140, 255))
            return image

    @classmethod
    def unnormalize(cls, img):
        img = img.clone()
        for im_channel, mean, std in zip(img, cls.mean_img, cls.std_img):
            im_channel.mul_(std).add_(mean)
        return img