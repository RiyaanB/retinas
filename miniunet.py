import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import image
import numpy as np
from random import seed


class RGBDataset(Dataset):
    def __init__(self, img_dir):
        """
            Initialize instance variables.
            :param img_dir (str): path of train or test folder.
            :return None:
        """
        # TODO: complete this method
        # ===============================================================================
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.img_dir = img_dir

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean_rgb, std_rgb),
            ])

        self.dataset_length = 0
        for file in os.listdir(img_dir + "rgb"):
            if file.endswith(".png"):
                self.dataset_length += 1
        # ===============================================================================

    def __len__(self):
        """
            Return the length of the dataset.
            :return dataset_length (int): length of the dataset, i.e. number of samples in the dataset
        """
        # TODO: complete this method
        # ===============================================================================
        return self.dataset_length
        # ===============================================================================

    def __getitem__(self, idx):
        """
            Given an index, return paired rgb image and ground truth mask as a sample.
            :param idx (int): index of each sample, in range(0, dataset_length)
            :return sample: a dictionary that stores paired rgb image and corresponding ground truth mask.
        """
        # TODO: complete this method
        # Hint:
        # - Use image.read_rgb() and image.read_mask() to read the images.
        # - Think about how to associate idx with the file name of images.
        # - Remember to apply transform on the sample.
        # ===============================================================================
        rgb_img = self.transform(image.read_rgb(self.img_dir + f'rgb/{idx}_rgb.png'))
        gt_mask = torch.as_tensor(image.read_mask(self.img_dir + f'gt/{idx}_gt.png'), dtype=torch.long)
        sample = {'input': rgb_img, 'target': gt_mask}
        return sample
        # ===============================================================================

x


class MiniUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super(MiniUNet, self).__init__()
        # ===============================================================================
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conv1  = nn.Conv2d(n_channels, 16, 3, padding=1)
        self.conv2  = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3  = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4  = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5  = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6  = nn.Conv2d(128+256, 128, 3, padding=1)
        self.conv7  = nn.Conv2d(64+128, 64, 3, padding=1)
        self.conv8  = nn.Conv2d(32+64, 32, 3, padding=1)
        self.conv9  = nn.Conv2d(16+32, 16, 3, padding=1)
        self.conv10 = nn.Conv2d(16, n_classes, 1)
        # ===============================================================================

    def forward(self, x):
        # ===============================================================================
        F = nn.functional

        output1 = F.relu(self.conv1(x))
        output2 = F.relu(self.conv2(F.max_pool2d(output1, (2,2))))
        output3 = F.relu(self.conv3(F.max_pool2d(output2, (2,2))))
        output4 = F.relu(self.conv4(F.max_pool2d(output3, (2,2))))
        output5 = F.relu(self.conv5(F.max_pool2d(output4, (2,2))))
        output6 = F.interpolate(output5, scale_factor=2)
        input7 = torch.cat((output4, output6), dim=1)
        output7 = F.interpolate(F.relu(self.conv6(input7)), scale_factor=2)
        input8 = torch.cat((output3, output7), dim=1)
        output8 = F.interpolate(F.relu(self.conv7(input8)), scale_factor=2)
        input9 = torch.cat((output2, output8), dim=1)
        output9 = F.interpolate(F.relu(self.conv8(input9)), scale_factor=2)
        input10 = torch.cat((output1, output9), dim=1)
        output = self.conv10(self.conv9(input10))
        return output
        # ===============================================================================
