import copy
import math
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils import data as torch_data
from torch.nn import functional as F
from torch.distributed.nn.functional import all_gather as all_gather_with_backprop
from torch_scatter import scatter_min

from torchdrug import core, layers, models, tasks, metrics
from torchdrug.data import constant, protein
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.utils import comm

from torchmetrics.image.fid import FrechetInceptionDistance


import ipdb 

@R.register("tasks.PriorReweighing")
class PriorReweighing(tasks.Task, core.Configurable):

    def __init__(self, vae_model, adv_model, sample_ratio=1.):
        super(PriorReweighing, self).__init__()
        self.vae_model = vae_model
        self.adv_model = adv_model
        self.sample_ratio = sample_ratio

        self.vae_model.eval()
        self.vae_model.requires_grad_(False)
    
    def predict_and_target(self, batch, all_loss=None, metric=None):
        pos_z = self.sample_from_q(batch)
        neg_z = self.sample_from_p(int(len(batch) * self.sample_ratio))

        zs = torch.cat([pos_z, neg_z], dim=0)
        input = self.vae_model.decode(zs)
        
        preds = self.adv_model(input)
        #ipdb.set_trace()
        
        targets = torch.cat([
                    torch.ones(len(pos_z), dtype=torch.long, device=self.device),
                    torch.zeros(len(neg_z), dtype=torch.long, device=self.device)],
                    dim=0
                )
        return preds, targets
    
    def evaluate(self, pred, target):
        metric = {}

        acc_name = tasks._get_metric_name("acc")
        accuracy = ((pred.squeeze() > 0.5).long() == target).float().mean()
        metric[acc_name] = accuracy

        return metric

    def get_loss(self, pred, target, all_loss, metric):
        ce_name = tasks._get_criterion_name("ce")
        loss = F.binary_cross_entropy(pred.squeeze(), target.float())
        metric[ce_name] = loss
        all_loss += loss

        return all_loss, metric
    
    def forward(self, batch):

        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        
        pred, target = self.predict_and_target(batch, all_loss, metric)

        metric.update(self.evaluate(pred, target))

        all_loss, metric = self.get_loss(pred, target, all_loss, metric)

        return all_loss, metric
    
    @torch.no_grad()
    def get_reweight(self, z):
        self.adv_model.eval()
        x = self.vae_model.decode(z)
        D = self.adv_model(x)
        r = D / (1  - D)
        return r
    
    @torch.no_grad()
    def sample(self, num_samples):
        '''sampling-importance-resampling (SIR)'''
        z = self.sample_from_p(num_samples*10)
        r = self.get_reweight(z).reshape(-1,)
        r /= r.sum()
        idx = torch.multinomial(r, num_samples)

        return self.vae_model.decode(z[idx])

    @torch.no_grad()
    def sample_from_p(self, num_samples):
        z = torch.randn(num_samples, self.vae_model.latent_dim, device=self.device)
        return z

    @torch.no_grad()
    def sample_from_q(self, input):
        self.vae_model.eval()
        mu, logvar = self.vae_model.encode(input)
        z = self.vae_model.reparameterize(mu, logvar)
        return z
    

@R.register("tasks.ImageGeneration")
class ImageGeneration(tasks.Task, core.Configurable):

    def __init__(self, model, kl_weight=1.):
        super(ImageGeneration, self).__init__()
        self.model = model
        self.kl_weight = kl_weight
        self.fid = FrechetInceptionDistance(feature=2048)
    
    def predict_and_target(self, batch, all_loss=None, metric=None):
        outputs, mus, logvars = self.model(batch)
        preds = {"output": outputs, "mu": mus, "logvar": logvars}
        targets = {"input": batch}
        return preds, targets
    
    def evaluate(self, pred, target):
        metric = {}

        batch_size = 512

        imgs_dist1 = (pred["output"].mul(255).add(0.5).clamp(0, 255)).type(torch.uint8)
        imgs_dist2 = (target["input"].mul(255).add(0.5).clamp(0, 255)).type(torch.uint8)
        if imgs_dist1.shape[1] == 1:
            imgs_dist1 = imgs_dist1.repeat(1, 3, 1, 1)
        if imgs_dist2.shape[1] == 1:
            imgs_dist2 = imgs_dist2.repeat(1, 3, 1, 1)
        
        #ipdb.set_trace()
        self.fid.reset()
        for idx in range(0, len(pred["output"]), batch_size):
            self.fid.update(imgs_dist1[idx:idx+batch_size], real=False)
            self.fid.update(imgs_dist2[idx:idx+batch_size], real=True)
        print("fake_samples: ", self.fid.fake_features_num_samples)
        print("real_samples: ", self.fid.real_features_num_samples)
        metric["FID"] = self.fid.compute()

        return metric
    
    def get_loss(self, pred, target, all_loss, metric):

        # reconstruction
        #loss = F.mse_loss(pred["output"], target["input"])
        #mse_name = tasks._get_criterion_name("mse")
        #metric["recons." + mse_name] = loss
        #ipdb.set_trace()
        N = len(pred["output"])
        ce_name = tasks._get_criterion_name("ce")
        loss = F.binary_cross_entropy(pred["output"].reshape(N, -1), target["input"].reshape(N, -1), reduction='sum') / N
        metric["recons." + ce_name] = loss
        all_loss += loss
        # KL divergence
        loss = torch.mean(-0.5 * torch.sum(1 + pred["logvar"] 
                          - pred["mu"] **2 - pred["logvar"].exp(), dim=1), dim=0)
        metric["KL"] = loss
        all_loss += loss * self.kl_weight

        return all_loss, metric
    
    def forward(self, batch):

        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        #metric.update(self.evaluate(pred, target))

        all_loss, metric = self.get_loss(pred, target, all_loss, metric)

        return all_loss, metric
    
    @torch.no_grad()
    def sample(self, num_samples):
        return self.model.sample(num_samples)

    def generate(self, x):
        return self.model(x)[0]