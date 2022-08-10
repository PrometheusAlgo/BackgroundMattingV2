import argparse
import torch
import os, time
import numpy as np
from PIL import Image
from threading import Thread
from torchvision.transforms.functional import to_pil_image

from model import MattingBase, MattingRefine
# from inference_utils import HomographicAlignment

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# Worker function
def writer(img, path):
    img = to_pil_image(img[0].cpu())
    img.save(path)

# helper
class BgrMattingHelper:
    def __init__ (self, args):
        self.device = DEVICE
        self.stream = None

        # defaults
        if not hasattr(args, 'gpu'):
            args.gpu = 0
        if not hasattr(args, 'model_type'):
            args.model_type = "mattingrefine"
        if not hasattr(args, 'model_backbone'):
            args.model_backbone = "resnet101"
        if not hasattr(args, 'model_backbone_scale'):
            args.model_backbone_scale = 0.25
        if not hasattr(args, 'model_refine_mode'):
            args.model_refine_mode = 'sampling'
        if not hasattr(args, 'model_refine_sample_pixels'):
            args.model_refine_sample_pixels = 80_000
        if not hasattr(args, 'model_refine_threshold'):
            args.model_refine_threshold = 0.7
        if not hasattr(args, 'model_refine_kernel_size'):
            args.model_refine_kernel_size = 3
        
        print('[BgrMattingHelper] work on gpu %d' % args.gpu)
        print('[BgrMattingHelper] model_type =', args.model_type)
        print('[BgrMattingHelper] model_backbone =', args.model_backbone)
        print('[BgrMattingHelper] model_backbone_scale =', args.model_backbone_scale)
        print('[BgrMattingHelper] model_refine_mode =', args.model_refine_mode)
        print('[BgrMattingHelper] model_refine_sample_pixels =', args.model_refine_sample_pixels)
        print('[BgrMattingHelper] model_refine_threshold =', args.model_refine_threshold)
        print('[BgrMattingHelper] model_refine_kernel_size =', args.model_refine_kernel_size)

        self.model = MattingRefine(
            args.model_backbone,
            args.model_backbone_scale,
            args.model_refine_mode,
            args.model_refine_sample_pixels,
            args.model_refine_threshold,
            args.model_refine_kernel_size)

        torch.cuda.set_device(int(args.gpu))
        self.model = self.model.to(DEVICE).eval()
        self.model.load_state_dict(torch.load(args.model_checkpoint, map_location=DEVICE), strict=False)

    def calcBackgroundMatting(self, src, bgr):
        with torch.no_grad():
            start = time.time()

            src_clr = src[:, :3, :, :]
            bgr_clr = bgr[:, :3, :, :]

            pha, fgr, _, _, err, ref = self.model(src_clr, bgr_clr)
            thresh = 1. / 255.
            fgr[(pha < thresh).repeat([1, 3, 1, 1])] = 0

            fgr = torch.concat([fgr, pha], dim=1)
            # if src.shape[1] == 4:
            #     invalid_idx = (src[:, 3:4, :, :] < 0.999)
            #     # fgr[:, 3:4, :, :][invalid_idx] = src[:, 3:4, :, :][invalid_idx]
            #     fgr[:, 3:4, :, :][invalid_idx] = 0

            end = time.time()
            print('[BgrMattingHelper] elapsed =', end-start)

        return pha, fgr

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
