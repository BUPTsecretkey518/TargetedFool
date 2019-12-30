# coding:utf-8
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
from targetedfool import targetedfool



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

net = torch.load('/model/net_008.pkl')

# Switch to evaluation mode
net.eval()
s_time = 0
s_r = 0
p_adv = 0
t = 2
for i in range(101):

    im_orig = Image.open('/test/%d.png' % i)
    mean = [0.5, ]
    std = [0.5, ]

    im1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])(im_orig)

    r, loop_i, label_orig, label_pert, pert_image, elapsed = targetedfool(im1, net, t)

    print('time: ', elapsed)
    s_time = s_time + elapsed

    print("Original label = ", label_orig)
    print("Perturbed label = ", label_pert)

    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv * torch.ones(A.shape))
        A = torch.min(A, maxv * torch.ones(A.shape))
        return A
    clip = lambda x: clip_tensor(x, 0, 255)

    tf = transforms.Compose([
        transforms.Normalize(mean=[0, ], std=list(map(lambda x: 1 / x, std))),
        transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, ]),
        transforms.ToPILImage(),
        transforms.CenterCrop(28)
    ])

    unloader = transforms.ToPILImage()
    def tensor_to_PIL(tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image

    # tf(im1).save('lenet_out/%d_1.jpg'%i)
    # tf(pert_image.cpu()[0]).save('lenet_out/%d_2.jpg'%i)
    im_2 = np.linalg.norm(im1, ord=None, axis=None, keepdims=False)
    r_2 = np.linalg.norm(pert_image.cpu()[0] - im1, ord=None, axis=None, keepdims=False)
    adv = r_2 / im_2
    s_r = s_r + r_2
    p_adv = p_adv + adv
    # tensor_to_PIL((pert_image.cpu()[0]-im1)).save('img/%d_3.jpg'%i)

print("s_time: ", s_time / 100)
print("r: ", s_r / 100)
print("p_adv: ", p_adv / 100)