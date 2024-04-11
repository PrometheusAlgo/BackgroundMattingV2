
import os, time
import torch
from . import MattingBase, MattingRefine

# helper
class BgrMattingHelper:
    def __init__ (self, args, device='cuda', verbose=False):
        self.device = device
        self.verbose = verbose

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
        if not hasattr(args, 'model_checkpoint'):
            args.model_checkpoint = os.path.abspath(os.path.join(__file__, f'../../models/pytorch/pytorch_{args.model_backbone}.pth'))
        
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
        self.model = self.model.to(self.device).eval()
        self.model.load_state_dict(torch.load(args.model_checkpoint, map_location=self.device), strict=False)

    def calcBackgroundMatting(self, src, bgr):
        with torch.no_grad():
            if self.verbose:
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

            if self.verbose:
                end = time.time()
                print('[BgrMattingHelper] elapsed =', end-start)

        return pha, fgr
    
