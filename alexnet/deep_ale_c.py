#coding:utf-8
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
from Python.target_deepfool import target_deepfool



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),   # (32-3+2)/1+1 = 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (32-2)/2+1 = 16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (16-2)/2+1 = 8
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # (8-3+2)/1+1 = 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (8-2)/2+1 = 4
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # (4-3+2)/1+1 = 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (4-2)/2+1 = 2
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),# (2-3+2)/1+1 = 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (2-2)/2+1 = 1
        )                            # output 1 * 1 * 128

        self.dense = nn.Sequential(
            nn.Linear(128, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 128)
        x = self.dense(x)
        return x

net = torch.load('model.pkl')
# Switch to evaluation mode
net.eval()
t = 1
p = 9
s_time=0
s_r=0
p_adv=0
for t in range(0,1,1):
    if t != p:
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        for i in range(1,11,1):
            im_orig = Image.open('images/truck/%d.png'%i)
            im1 = transforms.Compose([
                    transforms.ToTensor(),
            ])(im_orig)

            r, loop_i, label_orig, label_pert, pert_image,t,elapsed = target_deepfool(im1, net,t)
            print('time: ', elapsed)
            s_time = s_time + elapsed

            print("Original label = ",label_orig, classes[label_orig])
            print("Perturbed label = ", label_pert ,classes[label_pert])

            def clip_tensor(A, minv, maxv):
                A = torch.max(A, minv*torch.ones(A.shape))
                A = torch.min(A, maxv*torch.ones(A.shape))
                return A

            clip = lambda x: clip_tensor(x, 0, 255)
            tf = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.CenterCrop(28)
                                     ])

            unloader = transforms.ToPILImage()
            def tensor_to_PIL(tensor):
                image = tensor.cpu().clone()
                image = image.squeeze(0)
                image = unloader(image)
                return image

            if label_orig != t:
                im_2 = np.linalg.norm(im1, ord=None, axis=None, keepdims=False)
                r_2 = np.linalg.norm(pert_image.cpu()[0] - im1, ord=None, axis=None, keepdims=False)
                adv = r_2 / im_2
                s_r = s_r + r_2
                p_adv = p_adv + adv

        print("s_time: ",s_time/10)
        print("r: ", s_r / 10)
        print("p_adv: ", p_adv / 10)