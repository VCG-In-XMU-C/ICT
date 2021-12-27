# Inference

from utils import util
import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import logging
from utils.util import set_seed
from models.model import GPTConfig,GPT
import argparse
from utils.util import sample_mask,sample_mask_all
from tqdm import tqdm
from PIL import Image
import os
import time
import torchvision.transforms as transforms

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='ICT', help='The name of this exp')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--gpus', type=str, default=[0, 1])
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--BERT', action='store_true', help='BERT model, Image Completion')
    parser.add_argument('--image_url', type=str, default='D:\\Data\\FaceScape_dist_list\\test\\',
                        help='the folder of image')
    parser.add_argument('--mask_url', type=str, default='D:\\Data\\FaceScape_dist_list\\masks\\',
                        help='the folder of mask')
    parser.add_argument('--top_k', type=int, default=100)

    parser.add_argument('--image_size', type=int, default=256, help='input sequence length: image_size*image_size')

    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=1024)
    parser.add_argument('--GELU_2', action='store_true', help='use the new activation function')

    parser.add_argument('--save_url', type=str, default='./results/', help='save the output results')
    parser.add_argument('--n_samples', type=int,default=8, help='sample cnt')

    parser.add_argument('--sample_all', action='store_true', help='sample all pixel together, ablation use')
    parser.add_argument('--skip_number', type=int, default=0,
                        help='since the inference is slow, skip the image which has been inferenced')

    parser.add_argument('--no_progressive_bar', action='store_true', help='')
    # parser.add_argument('--data_path',type=str,default='/home/ziyuwan/workspace/data/')

    opts = parser.parse_args()

    s_time = time.time()

    # model_config=GPTConfig(512,32*32,
    #                        embd_pdrop=0.0, resid_pdrop=0.0, 
    #                        attn_pdrop=0.0, n_layer=14, n_head=8,
    #                        n_embd=256,BERT=opts.BERT)

    model_config = GPTConfig(16, embd_pdrop=0.0, resid_pdrop=0.0,
                             attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head,
                             n_embd=opts.n_embd, BERT=opts.BERT, use_gelu2=opts.GELU_2)

    # Load model
    IGPT_model = GPT(model_config)
    ckpt_path = os.path.join(opts.ckpt_path, opts.name, 'best.pth')
    checkpoint = torch.load(ckpt_path)
    
    if opts.ckpt_path.endswith('.pt'):
        IGPT_model.load_state_dict(checkpoint)
    else:
        IGPT_model.load_state_dict(checkpoint['model'])

    IGPT_model.cuda()

    n_samples = opts.n_samples

    img_list = sorted(os.listdir(opts.image_url))
    mask_list = sorted(os.listdir(opts.mask_url))
    # mask_list=mask_list[-len(img_list):]
    if opts.skip_number > 0:
        img_list = img_list[opts.skip_number-1:]
        mask_list = mask_list[opts.skip_number-1:]
        print("Resume from %s" % (img_list[0]))

    gray_transforms = transforms.Compose([
        transforms.Resize((opts.image_size, opts.image_size), Image.BILINEAR),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    normalize = transforms.Normalize((0.5,), (0.5,))

    for img_name in img_list:
        for mask_name in mask_list:
            image_url = os.path.join(opts.image_url, img_name)
            input_image = Image.open(image_url).convert("L")
            x = gray_transforms(input_image)
            x = normalize(x)

            mask_url = os.path.join(opts.mask_url, mask_name)
            input_mask = Image.open(mask_url).convert("L")
            mask = gray_transforms(input_mask)
            mask[mask > 0.99] = 1
            mask[mask != 1] = 0
            masked = mask * x
            x = x.reshape(1, 1, opts.image_size, opts.image_size).cuda()
            mask = mask.reshape(1, 1, opts.image_size, opts.image_size).cuda()
            masked = masked.reshape(1, 1, opts.image_size, opts.image_size).cuda()

            fake, loss = IGPT_model(masked, x)

            current_url = os.path.join(opts.save_url, opts.name)
            os.makedirs(current_url, exist_ok=True)
            prefix = img_name.replace('png', '')
            suffix = mask_name.replace('png', '')
            path_str = current_url + '/' + prefix + '_' + suffix + '_'

            im_fake = util.tensor2im(fake)
            im_masked = util.tensor2im(masked)
            im_x = util.tensor2im(x)
            im_mask = util.tensor2im(mask)
            util.save_image(im_fake, path_str + 'im_fake.png', aspect_ratio=1)
            util.save_image(im_masked, path_str + 'im_input.png', aspect_ratio=1)
            util.save_image(im_x, path_str + 'im_gt.png', aspect_ratio=1)
            util.save_image(im_mask, path_str + 'im_mask.png', aspect_ratio=1)

            # for i in range(n_samples):
            #
            #     current_url=os.path.join(opts.save_url,'condition_%d'%(i+1))
            #     os.makedirs(current_url,exist_ok=True)
            #     current_img=C[pixels[i]].view(opts.image_size, opts.image_size, 3).numpy().astype(np.uint8)
            #     tmp=Image.fromarray(current_img)
            #     tmp.save(os.path.join(current_url,img_name))
            print("Finish %s" % img_name)
        
        e_time = time.time()
        print("This test totally costs %.5f seconds" % (e_time-s_time))
