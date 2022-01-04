import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import joblib


logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 8
    # n_embd = 768
    n_embd = 32*4


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                      .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

        self.config=config

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # if not self.config.BERT:
        #     att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class Block_2(nn.Module):
    """ Transformer block with original GELU2 """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU2(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Block(nn.Module):
    """ Transformer block with original GELU """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # TODO: Something should be noted here: 
        # 1. The learning of SOS token. [x]
        # 2. The learning of extra embedding, which is used to inner product to generate the final probability [x]

        # Start token
        # if not config.BERT:
        #     self.sos = torch.nn.Parameter(torch.zeros(config.n_embd))
        #     nn.init.normal_(self.sos)

        self.NUM_PCA_COMPONENTS = 32

        # input embedding stem
        self.tok_emb = nn.Linear(self.NUM_PCA_COMPONENTS, config.n_embd, bias=False)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        if config.use_gelu2:
            self.blocks = nn.Sequential(*[Block_2(config) for _ in range(config.n_layer)])
        else:
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, self.NUM_PCA_COMPONENTS, bias=False)

        self.block_size = config.block_size
        self.config = config

        self.apply(self._init_weights)

        self.criterionL1 = torch.nn.L1Loss()

        self.pca_model = joblib.load('.\\pca_%d.m' % self.NUM_PCA_COMPONENTS)  # load trained pca model
        self.m_pca_model = joblib.load('.\\m_pca_%d.m' % self.NUM_PCA_COMPONENTS)  # load trained pca model
        self.pca_components = torch.from_numpy(self.pca_model.components_).cuda()
        self.pca_mean = torch.from_numpy(self.pca_model.mean_).cuda()
        # self.pca_inverse = self.pca_components.T.cuda()
        self.m_pca_components = torch.from_numpy(self.m_pca_model.components_).cuda()
        self.m_pca_mean = torch.from_numpy(self.m_pca_model.mean_).cuda()
        self.m_pca_inverse = self.m_pca_components.T.cuda()

        logger.info("number of parameters: %f MB", sum(p.numel() for p in self.parameters())/1024/1024)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        # if not self.config.BERT:
        #     no_decay.add('sos')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, input_image, targets=None, masks=None):
        # print(input_image.shape)
        # print(targets.shape)
        # shape of input image: b, 1, 256, 256
        # device = input_image.device
        data = rearrange(input_image, 'b 1 (h p1) (w p2) -> (b h w) (p1 p2)', p1=8, p2=8)
        # target_emb = rearrange(targets, 'b 1 (h p1) (w p2) -> (b h w) (p1 p2)', p1=64, p2=64)
        # shape of data: b * 16, 4096

        data = torch.mm(data - self.m_pca_mean, self.m_pca_inverse)
        # data = torch.from_numpy(self.pca_model.transform(data.cpu())).to(device)
        # target_emb = torch.from_numpy(self.pca_model.transform(target_emb.cpu())).to(device)
        # target_emb = target_emb.type(torch.float32)
        # data = data.type(torch.float32)
        # shape of coffs: b * 16, 512
        # target_emb = rearrange(target_emb, '(b n) p -> b n p', n=16)
        data = rearrange(data, '(b n) p -> b n p', n=32*32)

        b, t, s = data.size()
        # assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(data) # each index maps to a (learnable) vector
        # token_embeddings = torch.cat([sos, token_embeddings[:, :-1, :]], axis=1)

        # if self.config.BERT:
        #     masks = masks.unsqueeze(2)
        #     token_embeddings = token_embeddings* (1 - masks)
        # else:
        # sos: start of sentence
        # sos = torch.ones(b, 1, self.config.n_embd, device=data.device) * self.sos
        # token_embeddings = torch.cat([sos, token_embeddings[:, :-1, :]], axis=1)

        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        # shape of logits: b, 16, 512
        # logits = rearrange(logits, 'b 16 512 -> (b 16) 512')
        fake = rearrange(logits, 'b n p -> (b n) p', n=32*32)
        # shape of logits: b * 16, 512
        fake = torch.mm(fake, self.pca_components) + self.pca_mean
        # fake = torch.from_numpy(self.pca_model.inverse_transform(fake.detach().cpu())).to(device)
        # fake = fake.type(torch.float32)
        # shape of logits: b * 16, 4096
        fake = rearrange(fake, '(b h w) (p1 p2) -> b 1 (h p1) (w p2)', w=32, h=32, p1=8)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # if self.config.BERT:
            #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),reduce=False)
            #
            #     # if torch.isnan(loss).any():
            #     #     print("###########Warning, this iteration appears NAN###########")
            #     #     print(idx)
            #     #     print(targets)
            #     #     print("#######################################################")
            #     masks = masks.view(-1)
            #     loss *= masks
            #     if not self.config.dynamic_weight:
            #         loss = torch.mean(loss)
            #     else:
            #         loss = torch.sum(loss) / torch.sum(masks)
            # else:
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = self.criterionL1(fake, targets)

        return fake, loss
