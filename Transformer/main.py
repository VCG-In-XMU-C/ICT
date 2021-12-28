import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import logging
from utils.util import set_seed, Logger
from datas.dataset import FaceScapeDataset
from models.model import GPTConfig, GPT
from DDP_trainer import TrainerConfig, Trainer
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
    train_dataset = FaceScapeDataset(opts.data_path, mask_path=opts.mask_path,
                                     is_train=True, image_size=opts.image_size)
    test_dataset = FaceScapeDataset(opts.validation_path, mask_path=opts.mask_path,
                                    is_train=False, image_size=opts.image_size)

    # vocab_size指的是每个token分多少类（有多少个词备选），在我们的任务里不需要，我们不分类的
    # 对了，回归任务不用交叉熵，分类任务才用，回归用L1就好
    model_config = GPTConfig(train_dataset.block_num, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                             n_layer=opts.n_layer, n_head=opts.n_head, n_embd=opts.n_embd, BERT=False,
                             use_gelu2=opts.GELU_2, dynamic_weight=opts.dynamic_weight)

    # Original n_layer=12, n_head=8, n_embd=256
    IGPT_model = GPT(model_config)

    tokens_per_epoch = len(train_dataset.image_id_list)*train_dataset.block_num
    
    train_epochs = opts.train_epoch

    # By default: 8xV100 GPUs
    # TODO: Modify the ckpt path [√]
    train_config = TrainerConfig(max_epochs=train_epochs, batch_size=opts.batch_size,
                                 learning_rate=opts.lr, betas=(0.9, 0.95),
                                 weight_decay=0, lr_decay=True, warmup_tokens=tokens_per_epoch,
                                 final_tokens=train_epochs*tokens_per_epoch, ckpt_path=opts.ckpt_path,
                                 num_workers=8, GPU_ids=opts.gpu, BERT=False, world_size=1,
                                 AMP=opts.AMP, print_freq=opts.print_freq)
    trainer = Trainer(IGPT_model, train_dataset, test_dataset, train_config, gpu, opts.gpus)
    loaded_ckpt = trainer.load_checkpoint(opts.resume_ckpt)
    trainer.train(loaded_ckpt)
    print("Finish the training ...")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='ICT', help='The name of this exp')
    parser.add_argument('--gpus', type=str, default=[0, 1])
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--data_path', type=str, default='D:\\Data\\FaceScape_dist_list\\train\\',
                        help='Indicate where is the training set')
    parser.add_argument('--mask_path', type=str, default='D:\\Data\\FaceScape_dist_list\\masks\\')
    parser.add_argument('--BERT', action='store_true', help='Use bert objective to train')
    parser.add_argument('--ImageNet', action='store_true', help='Training with ImageNet')
    parser.add_argument('--batch_size', type=int, default=1, help='16*8 maybe suitable for V100')
    parser.add_argument('--train_epoch', type=int, default=80, help='how many epochs')
    parser.add_argument('--print_freq', type=int, default=200, help='While training, the freq of printing log')

    parser.add_argument('--validation_path', type=str, default='D:\\Data\\FaceScape_dist_list\\test\\',
                        help='where is the validation set of ImageNet')

    parser.add_argument('--image_size', type=int, default=256, help='input sequence length = image_size*image_size')

    # Define the size of transformer
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--GELU_2', action='store_true', help='use the new activation function')

    parser.add_argument('--random_stroke', action='store_true', help='use the generated mask')

    # Adjust the objective weight of log-likelihood
    parser.add_argument('--dynamic_weight', action='store_true',help='Not mean directly, based on the mask regions')

    parser.add_argument('--use_ImageFolder', action='store_true',help='using the original folder for ImageNet dataset')

    # DDP+AMP
    # parser.add_argument('--DDP', action='store_true',help='using DDP rather than normal data parallel')
    # parser.add_argument('--nodes', type=int,default=1,help='how many machines')
    # parser.add_argument('--gpus', type=int, default=1, help='how many GPUs in one node')
    # parser.add_argument('--node_rank',type=int,default=0,help='the id of this machine')
    parser.add_argument('--AMP', action='store_true', help='Automatic Mixed Precision')
    parser.add_argument('--resume_ckpt', type=str, default='latest.pth', help='start from where, the default is latest')

    opts = parser.parse_args()

    opts.BERT = False

    opts.ckpt_path = os.path.join(opts.ckpt_path, opts.name)
    opts.resume_ckpt = os.path.join(opts.ckpt_path, opts.resume_ckpt)
    os.makedirs(opts.ckpt_path, exist_ok=True)

    # opts.world_size=opts.nodes*opts.gpus
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '48364'

    logging.basicConfig(
            # filename=os.path.join(opts.ckpt_path,'running.log'),
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )

    main_worker(opts.gpu, opts)
    # mp.spawn(main_worker, nprocs=opts.gpus, args=(opts,))
