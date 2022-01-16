
## Inference
import cv2
import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import logging
from utils.util import set_seed
from models.model import GPTConfig, GPT
import argparse
from utils.util import sample_mask, sample_mask_all
from tqdm import tqdm
from PIL import Image
import os
import time
from skimage.measure import compare_psnr as psnr
from skimage import measure


def save_result(path, result):
    tmp = open(path, mode='w')
    tmp.write(result)
    tmp.close()


def calc_ssim_psnr(img1, img2):
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # img1[img2 == 0] = 0
    # return ssim(img1, img2), psnr(img1, img2)
    return measure.compare_ssim(img1, img2), psnr(img1, img2)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='ICT', help='The name of this exp')
    parser.add_argument('--GPU_ids', type=str, default='1')
    parser.add_argument('--ckpt_path', type=str, default='/mnt/datadisk0/Transformer/')
    parser.add_argument('--BERT', action='store_true', help='BERT model, Image Completion')
    parser.add_argument('--image_url', type=str, default='/mnt/datadisk0/final/test/images/',
                        help='the folder of image')
    parser.add_argument('--mask_url', type=str, default='/mnt/datadisk0/final/small_masks/',
                        help='the folder of mask')
    parser.add_argument('--top_k', type=int, default=20)

    parser.add_argument('--image_size', type=int, default=32, help='input sequence length: image_size*image_size')

    parser.add_argument('--n_layer', type=int, default=14)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--GELU_2', action='store_true', help='use the new activation function')

    parser.add_argument('--save_url', type=str, default='./result', help='save the output results')
    parser.add_argument('--n_samples', type=int, default=4, help='sample cnt')

    parser.add_argument('--sample_all', action='store_true', help='sample all pixel together, ablation use')
    parser.add_argument('--skip_number', type=int, default=0,
                        help='since the inference is slow, skip the image which has been inferenced')

    parser.add_argument('--no_progressive_bar', action='store_true', help='')

    parser.add_argument('--class_size', type=int, default=7, help='cls')
    # parser.add_argument('--data_path',type=str,default='/home/ziyuwan/workspace/data/')

    opts = parser.parse_args()

    s_time = time.time()

    # model_config=GPTConfig(512,32*32,
    #                        embd_pdrop=0.0, resid_pdrop=0.0, 
    #                        attn_pdrop=0.0, n_layer=14, n_head=8,
    #                        n_embd=256,BERT=opts.BERT)

    model_config = GPTConfig(256, opts.image_size*opts.image_size, embd_pdrop=0.0, resid_pdrop=0.0,
                             attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head,
                             n_embd=opts.n_embd, BERT=opts.BERT, use_gelu2=opts.GELU_2, class_size=opts.class_size)

    # Load model
    IGPT_model = GPT(model_config)
    ckpt_path = os.path.join(opts.ckpt_path, opts.name, 'best.pth')
    checkpoint = torch.load(ckpt_path)
    
    if opts.ckpt_path.endswith('.pt'):
        IGPT_model.load_state_dict(checkpoint)
    else:
        IGPT_model.load_state_dict(checkpoint['model'])

    IGPT_model.eval()
    IGPT_model.cuda()

    n_samples = opts.n_samples

    img_list = sorted(os.listdir(opts.image_url))
    mask_list = sorted(os.listdir(opts.mask_url))
    # mask_list=mask_list[-len(img_list):]
    if opts.skip_number > 0:
        img_list = img_list[opts.skip_number-1:]
        mask_list = mask_list[opts.skip_number-1:]
        print("Resume from %s" % (img_list[0]))

    if opts.BERT:
        # cls_list = []
        # target_list = []
        # m_list = []
        ssim_list = 0
        psnr_list = 0
        dataset_size = len(img_list) * len(mask_list)
        for x_name in img_list:
            for y_name in mask_list:
                # if x_name != y_name:
                #     print("### Something Wrong ###")

                image_url = os.path.join(opts.image_url, x_name)
                # m_list.append(int(y_name[-6:-4]))
                # target_list.append(int(image_url[-6:-4]))
                input_image = Image.open(image_url).convert("L")
                x = input_image.resize((opts.image_size, opts.image_size), resample=Image.BILINEAR)
                x = torch.from_numpy(np.array(x)).view(-1)

                mask_url = os.path.join(opts.mask_url, y_name)
                input_mask = Image.open(mask_url).convert("L")
                y = input_mask.resize((opts.image_size, opts.image_size), resample=Image.NEAREST)
                y = torch.from_numpy(np.array(y) / 255.).view(-1)
                y = y > 0.5
                y = y.float()

                # shape of x is [32 * 32, 3]
                # shape of y is [32 * 32]

                img_name = x_name[:-4] + '_' + y_name[:-4] + '.png'
                mask_url = os.path.join(opts.save_url, opts.name, 'masked')
                os.makedirs(mask_url, exist_ok=True)

                masked = x * y
                masked = masked.reshape(opts.image_size, opts.image_size).numpy().astype(np.uint8)
                masked = Image.fromarray(masked)
                masked.save(os.path.join(mask_url, img_name))

                raw_url = os.path.join(opts.save_url, opts.name, 'raw')
                os.makedirs(raw_url, exist_ok=True)
                raw = x.reshape(opts.image_size, opts.image_size).numpy().astype(np.uint8)
                raw = Image.fromarray(raw)
                raw.save(os.path.join(raw_url, img_name))

                a_list = [x] * n_samples
                a_tensor = torch.stack(a_list, dim=0)  # Input images
                b_list = [y] * n_samples
                b_tensor = torch.stack(b_list, dim=0)  # Input masks
                b_tensor = b_tensor.int()
                # a_tensor *= (1 - b_tensor)

                if opts.sample_all:
                    pixels = sample_mask_all(IGPT_model, context=a_tensor.long(), length=opts.image_size * opts.image_size,
                                             num_sample=n_samples, top_k=opts.top_k, mask=b_tensor,
                                             no_bar=opts.no_progressive_bar)
                else:
                    pixels = sample_mask(IGPT_model, context=a_tensor.long(), length=opts.image_size * opts.image_size,
                                         num_sample=n_samples, top_k=opts.top_k, mask=b_tensor,
                                         no_bar=opts.no_progressive_bar)

                # cls_list.append(cls[0].cpu())

                for i in range(n_samples):
                    current_url = os.path.join(opts.save_url, opts.name, 'condition_%d' % (i + 1))
                    os.makedirs(current_url, exist_ok=True)
                    current_img = pixels[i].view(opts.image_size, opts.image_size).cpu().numpy().astype(np.uint8)
                    if i == 0:
                        merge_url = os.path.join(opts.save_url, opts.name, 'merge')
                        os.makedirs(merge_url, exist_ok=True)
                        merge_img = pixels[i].cpu() * (1-y) + x * y
                        merge_img = merge_img.view(opts.image_size, opts.image_size).cpu().numpy().astype(np.uint8)
                        ssim_result, psnr_result = calc_ssim_psnr(merge_img, x.view(opts.image_size, opts.image_size).cpu().numpy().astype(np.uint8))
                        ssim_list += ssim_result
                        psnr_list += psnr_result
                        tmp1 = Image.fromarray(merge_img)
                        tmp1.save(os.path.join(merge_url, img_name))
                    tmp = Image.fromarray(current_img)
                    tmp.save(os.path.join(current_url, img_name))
                print("Finish %s" % img_name)

        cls_path = os.path.join(opts.save_url, opts.name)
        # np.save(os.path.join(cls_path, 'cls.npy'), cls_list)
        # np.save(os.path.join(cls_path, 'target.npy'), target_list)
        # np.save(os.path.join(cls_path, 'mask.npy'), m_list)

        # sub = np.subtract(cls_list, target_list)
        # sub = (sub == 0)
        # accuracy = sub.sum() / len(cls_list)

        average_ssim = ssim_list / dataset_size
        average_psnr = psnr_list / dataset_size
        result_dir = os.path.join(cls_path, 'result.txt')
        result_str = 'ssim = %f, psnr = %f' % (average_ssim, average_psnr)
        print(result_str)
        save_result(result_dir, result_str)

        e_time = time.time()
        print("This test totally costs %.5f seconds" % (e_time-s_time))
