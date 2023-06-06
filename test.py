# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import argparse
import os

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


def draw_bbox(preds, fname, imgs, obj_list, color_list, output_path):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


        cv2.imwrite(os.path.join(output_path, fname), imgs[i])



def main(args):
    start = time.time()

    test_path = args["test_path"]
    imgs_path = [os.path.join(test_path, name) for name in os.listdir(test_path)]
    output_path = args["output_path"]
    output_path = os.path.join(output_path, "images")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    compound_coef = args["compound_coef"]
    force_input_size = None  # set None to use default size

    # replace this part with your project's anchor config
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    threshold = 0.4
    iou_threshold = 0.5

    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = ["fod"]


    color_list = standard_to_bgr(STANDARD_COLORS)
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    fnames, orig_imgs, framed_imgs, framed_metas = [], [], [], []
    for path in imgs_path:
        fname = os.path.split(path)[1]
        orig_img, framed_img, framed_meta = preprocess(path, max_size=input_size)

        fnames.append(fname)
        orig_imgs.append(orig_img)
        framed_imgs.append(framed_img)
        framed_metas.append(framed_meta)

    batch = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    batch = batch.to(torch.float32).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(args["weights"]))
    model.requires_grad_(False)
    model.eval()
    model = model.cuda()

    with torch.no_grad():
        for x, fname, orig_img, framed_meta in zip(batch, fnames, orig_imgs, framed_metas):
            x = x.unsqueeze(0)
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

            out = invert_affine(framed_metas, out)
            draw_bbox(out, fname, orig_img, obj_list, color_list, output_path)

    end = time.time()
    print("Test Completed in: {:.2f} seconds".format(end - start))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="path to trained model")
    parser.add_argument("--test_path", type=str, default=None, help="path to test images")
    parser.add_argument("--output_path", type=str, default="output", help="output path")
    parser.add_argument("--compound_coef", type=int, default=2, help="Compound Coefficient")
    
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()
    args = vars(args)

    main(args)

