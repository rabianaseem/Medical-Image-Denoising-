import torch
import torch.nn as nn
from torch.nn import functional as F
from networks.JL_DCF import build_model
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
from torchvision.utils import save_image
import PIL.Image as pil_image
# from torchvision.utils import ssim
# import utils.pytorch_ssim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from datetime import datetime
import itertools
import nibabel as nib
from nibabel import processing
import gc

writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
size_gab = (160, 160)
size_coarse = (20, 20)
size_mid = (80, 80)


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.build_model()
        self.net.eval()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model))
        if config.mode == 'train':

            if self.config.load == '':
                'self.net = nn.DataParallel(self.net)
                self._model.model = torch.nn.DataParallel(self._model.model, device_ids=[2, 3])
                self.net = self.net.JLModule.load_pretrained_model(self.config.pretrained_model)  # load pretrained backbone

            else:
                self.net.load_state_dict(torch.load(self.config.load))  # load pretrained model
                torch.save(model.module.state_dict(), path)

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        
        self.net = build_model(self.config.arch)

        if torch.cuda.device_count() > 1:
            print(str(torch.cuda.device_count())+" GPUs detected!")
            self.net = nn.DataParallel(self.net)
            self.net = torch.nn.DataParallel(self.net)

        self.lr = self.config.lr
        self.wd = self.config.wd
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        self.print_network(self.net, 'JL-DCF Structure')

    def test(self):
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

                input = torch.cat((images, depth), dim=0)
                preds, pred_coarse, pred_mid  = self.net(input)
                preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 1000 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '_JLDCF.nii')    
                contrast_nif = nib.Nifti1Image(multi_fuse, np.eye(4))    
                nib.save(contrast_nif, filename)                       

        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        self.optimizer.zero_grad()
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_depth, sal_label = data_batch['sal_image'], data_batch['sal_depth'], data_batch[
                    'sal_label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:

                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label = sal_image.to(device), sal_depth.to(device), sal_label.to(device)

                sal_label_coarse = F.interpolate(sal_label, size_coarse, mode='bilinear', align_corners=True)
                sal_label_coarse = torch.cat((sal_label_coarse, sal_label_coarse), dim=0)
                sal_label_mid = F.interpolate(sal_label, size_mid, mode='bilinear', align_corners=True)
                sal_label_mid = torch.cat((sal_label_mid, sal_label_mid), dim=0)
                sal_input = torch.cat((sal_image, sal_depth), dim=0)
                sal_final, sal_coarse, sal_mid  = self.net(sal_input)
                
                criterion = nn.MSELoss(reduction='sum')
                sal_loss_mid = criterion(sal_mid, sal_label_mid)
                sal_loss_coarse = criterion(sal_coarse, sal_label_coarse)
                sal_loss_final = criterion(sal_final, sal_label)
                sal_loss_fuse = sal_loss_final + 1000 * sal_loss_coarse + 1000 * sal_loss_mid
                ssim_score = ssim(sal_label, sal_final, data_range=1000)
                ssim_score = 1 - ssim_score

                weighted_loss = (0.5 * sal_loss_fuse) + (0.5 * ssim_score)
                # sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                sal_loss = weighted_loss / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data
                sal_loss.backward()

                # adaumulate gradients as done in DSS
                aveGrad += 1
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if (i + 1) % (self.show_every // self.config.batch_size) == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                        epoch, self.config.epoch, i + 1, iter_num, r_sal_loss / (self.show_every / self.iter_size)))
                    # print('Learning rate: ' + str(self.lr))
                    writer.add_scalar('training loss', r_sal_loss / (self.show_every / self.iter_size),
                                      epoch * len(self.train_loader.dataset) + i)
                    r_sal_loss = 0

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)
        torch.save(model.module.state_dict(), path)

