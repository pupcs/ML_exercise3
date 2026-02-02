import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--output-dir', default='test_outputs')
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    hr_image = pil_image.open(args.image_file).convert('RGB')

    hr_width = (hr_image.width // args.scale) * args.scale
    hr_height = (hr_image.height // args.scale) * args.scale
    hr_image = hr_image.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
    
    lr_image = hr_image.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
    lr_image = lr_image.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
    name, ext = args.image_file.rsplit('.', 1)
    lr_image.save(f"{name}_lowRes_x{args.scale}.{ext}")
    
    lr_np = np.array(lr_image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(lr_np)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    hr_np = np.array(hr_image).astype(np.float32)
    hr_ycbcr = convert_rgb_to_ycbcr(hr_np)
    
    hr_y = hr_ycbcr[..., 0] / 255.0
    hr_y = torch.from_numpy(hr_y).to(device)
    hr_y = hr_y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = calc_psnr(hr_y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    name, ext = args.image_file.rsplit('.', 1)
    output.save(f"{name}_srcnn_x{args.scale}.{ext}")
