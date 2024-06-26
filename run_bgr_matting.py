import argparse
import torch
import os, time
import numpy as np
from PIL import Image
from threading import Thread
from torchvision.transforms.functional import to_pil_image
# from inference_utils import HomographicAlignment
from bgrmatv2.bgr_matting_helper import BgrMattingHelper

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# Worker function
def writer(img, path):
    img = to_pil_image(img[0].cpu())
    img.save(path)

if __name__ == '__main__':
    # --------------- Arguments ---------------
    parser = argparse.ArgumentParser(description='Inference images')
    parser.add_argument('--gpu', help="gpu index", type=int, default=0)
    parser.add_argument('--model-type', type=str, required=True, choices=['mattingbase', 'mattingrefine'])
    parser.add_argument('--model-backbone', type=str, required=True, choices=['resnet101', 'resnet50', 'mobilenetv2'])
    parser.add_argument('--model-backbone-scale', type=float, default=0.25)
    parser.add_argument('--model-checkpoint', type=str, required=True)
    parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
    parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
    parser.add_argument('--model-refine-threshold', type=float, default=0.7)
    parser.add_argument('--model-refine-kernel-size', type=int, default=3)

    parser.add_argument('--src-img', type=str, required=True)
    parser.add_argument('--bgr-img', type=str, required=True)

    # parser.add_argument('--preprocess-alignment', action='store_true')

    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--output-types', type=str, required=True, nargs='+', choices=['com', 'pha', 'fgr', 'err', 'ref'])

    args = parser.parse_known_args()[0]

    assert 'err' not in args.output_types or args.model_type in ['mattingbase', 'mattingrefine'], \
        'Only mattingbase and mattingrefine support err output'
    assert 'ref' not in args.output_types or args.model_type in ['mattingrefine'], \
        'Only mattingrefine support ref output'


    # --------------- Main ---------------
    bgr_mat = BgrMattingHelper(args)

    src = load_image(args.src_img)
    bgr = load_image(args.bgr_img)

    # 0~255 to 0~1
    src = src / 255.
    bgr = bgr / 255.

    if src.shape[1] == 3:
        src = torch.concat([src, torch.ones_like(src[:, :1, :, :])], dim=1)

    # src[:, 3:4, 0:100, :] = 0.5

    pha, fgr = bgr_mat.calcBackgroundMatting(src, bgr)

    for i in range(10):
        pha, fgr = bgr_mat.calcBackgroundMatting(src, bgr)

    Thread(target=writer, args=(pha, os.path.join(args.output_dir, 'pha.jpg'))).start()
    Thread(target=writer, args=(fgr, os.path.join(args.output_dir, 'fgr.png'))).start()
    
    pass
