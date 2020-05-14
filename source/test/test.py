import argparse
import os
import io
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import PIL.Image as pil_image
from model import BDRRN, BDRRN_cat
import glob
import math
import numpy
import cv2
from skimage.measure import compare_ssim as ssim


def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2)**2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='BDRRN')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--parts_dir', type=str, required=True)
    parser.add_argument('--gts_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--format', type=str, required=False, default='.png')
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--cuda_device', type=int, default=2)

    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)



    if opt.arch == 'BDRRN':
        model = BDRRN(opt.num_channels)
    elif opt.arch == 'BDRRN_cat':
        model = BDRRN_cat(opt.num_channels)
    else:
        raise KeyError()


    state_dict = model.state_dict()
    weights =  torch.load(opt.weights_path)['model_state_dict']
    model.load_state_dict(weights)
    #for n, p in weights:
    #    if n in state_dict.keys():
    #        state_dict[n].copy_(p)
    #    else:
    #        raise KeyError(n)

    cudnn.benchmark = True
    device_name = 'cuda:' + str(opt.cuda_device)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    avg_de = 0.0
    avg_te = 0.0

    ss_de = 0.0
    ss_te = 0.0

    images_path = glob.glob(os.path.join(opt.images_dir, '*' + opt.format))
    for image_path in images_path:
        filename = os.path.basename(image_path).split('.')[0]
        groundT = pil_image.open(os.path.join(opt.gts_dir,os.path.basename(image_path)))
        input = pil_image.open(image_path)#.convert('RGB')
        parts = pil_image.open(os.path.join(opt.parts_dir,os.path.basename(image_path)))
        size = numpy.array(input).shape

        if not numpy.array(groundT).shape == size:
          groundT = groundT.resize((size[1],size[0]), pil_image.BICUBIC )

        avg_de+= psnr(numpy.array(input), numpy.array(groundT))
        ss_de+= ssim(numpy.array(input), numpy.array(groundT),multichannel=True)

        #avg_de+= psnr(numpy.array(parts), numpy.array(groundT))
        #ss_de+= ssim(numpy.array(parts), numpy.array(groundT))


        input = transforms.ToTensor()(input).unsqueeze(0).to(device)
        parts = transforms.ToTensor()(parts).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input, parts)

        if opt.num_channels == 3:
            pred = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()

            output = pil_image.fromarray(pred, mode = 'RGB')
        else:
            pred = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy().squeeze()
            output = pil_image.fromarray(pred, mode = 'L')

        out_name = os.path.join(opt.outputs_dir, '{}{}'.format(filename,opt.format))

        cv2.imwrite(out_name,pred)

        avg_te += psnr(numpy.array(output), numpy.array(groundT))
        ss_te += ssim(numpy.array(output), numpy.array(groundT),multichannel=True)


    avg_de/=len(images_path)
    avg_te/=len(images_path)
    ss_de/=len(images_path)
    ss_te/=len(images_path)

    print('AvgDe Psnr:{:4f} \nAvgTe Psnr:{:4f}  \nDelta Psnr:{:4f}'.format(avg_de, avg_te, avg_te-avg_de))
    print('AvgDe SSIM:{:4f} \nAvgTe SSIM:{:4f}  \nDelta SSIM:{:4f}'.format(ss_de, ss_te, ss_te-ss_de))
