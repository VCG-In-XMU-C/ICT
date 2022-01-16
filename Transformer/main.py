import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import logging
from utils.util import set_seed,Logger
from datas.dataset import ImageNetDatasetMask
from models.model import GPTConfig,GPT
from DDP_trainer import TrainerConfig,Trainer
import argparse
import os
import sys

import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
import torch.multiprocessing as mp

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def main_worker(gpu, opts):

    torch.cuda.set_device(gpu)
    set_seed(42)
    sys.stdout = Logger(os.path.join(opts.ckpt_path, 'log.txt'))

    # Define the dataset
    train_dataset = ImageNetDatasetMask(opts.data_path, mask_path=opts.mask_path, is_train=True,
                                        use_ImageFolder=opts.use_ImageFolder, image_size=opts.image_size,
                                        random_stroke=opts.random_stroke)
    test_dataset = ImageNetDatasetMask(opts.validation_path, mask_path=opts.mask_path, is_train=False,
                                       use_ImageFolder=opts.use_ImageFolder, image_size=opts.image_size)

    model_config=GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                           embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head,
                           n_embd=opts.n_embd, BERT=opts.BERT, use_gelu2=opts.GELU_2,
                           dynamic_weight=opts.dynamic_weight, class_size=opts.class_size)

    # Original n_layer=12, n_head=8, n_embd=256
    IGPT_model = GPT(model_config)

    tokens_per_epoch = len(train_dataset.image_id_list)*train_dataset.block_size
    
    train_epochs = opts.train_epoch

    # By default: 8xV100 GPUs
    # TODO: Modify the ckpt path [√]
    train_config = TrainerConfig(max_epochs=train_epochs, batch_size=opts.batch_size,
                               learning_rate=opts.lr, betas=(0.9, 0.95),
                               weight_decay=0, lr_decay=True, warmup_tokens=tokens_per_epoch,
                               final_tokens=train_epochs*tokens_per_epoch, ckpt_path=opts.ckpt_path,
                               num_workers=8, GPU_ids=opts.GPU_ids, BERT=opts.BERT,
                               AMP=opts.AMP, print_freq=opts.print_freq)
    trainer = Trainer(IGPT_model, train_dataset, test_dataset, train_config, gpu)
    loaded_ckpt = trainer.load_checkpoint(opts.resume_ckpt)
    trainer.train(loaded_ckpt)
    print("Finish the training ...")


if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='ICT', help='The name of this exp')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, default='/mnt/datadisk0/Transformer/')
    parser.add_argument('--data_path', type=str, default='/mnt/datadisk0/final/train/images/',
                        help='Indicate where is the training set')
    parser.add_argument('--mask_path', type=str, default='/mnt/datadisk0/final/small_masks/')
    parser.add_argument('--BERT', action='store_true', help='Use bert objective to train')
    # parser.add_argument('--ImageNet', action='store_true', help='Training with ImageNet')
    parser.add_argument('--batch_size', type=int, default=16, help='16*8 maybe suitable for V100')  # todo
    parser.add_argument('--train_epoch', type=int, default=100, help='how many epochs')
    parser.add_argument('--print_freq', type=int, default=200, help='While training, the freq of printing log')

    parser.add_argument('--validation_path', type=str, default='/mnt/datadisk0/final/test/images/',
                        help='where is the validation set of ImageNet')

    parser.add_argument('--image_size', type=int, default=32, help='input sequence length = image_size*image_size')

    # Define the size of transformer
    parser.add_argument('--n_layer', type=int, default=14)  # try 12 14
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-4)
    # todo
    parser.add_argument('--GELU_2', action='store_true', help='use the new activation function')

    parser.add_argument('--random_stroke', action='store_true', help='use the generated mask')

    # Adjust the objective weight of log-likelihood
    parser.add_argument('--dynamic_weight', action='store_true', help='Not mean directly, based on the mask regions')
    parser.add_argument('--use_ImageFolder', action='store_true', help='using the original folder for ImageNet dataset')

    ### DDP+AMP
    parser.add_argument('--gpus', type=int, default=1, help='how many GPUs in one node')
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--AMP', action='store_true', help='Automatic Mixed Precision')
    parser.add_argument('--resume_ckpt', type=str, default='latest.pth', help='start from where, the default is latest')

    parser.add_argument('--class_size', type=int, default=7, help='cls')

    opts = parser.parse_args()
    opts.ckpt_path = os.path.join(opts.ckpt_path, opts.name)
    opts.resume_ckpt = os.path.join(opts.ckpt_path, opts.resume_ckpt)
    os.makedirs(opts.ckpt_path, exist_ok=True)

    logging.basicConfig(
            # filename=os.path.join(opts.ckpt_path,'running.log'),
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )
    main_worker(opts.gpu, opts)
