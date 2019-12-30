#coding:utf-8
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.models as models
import os
from PIL import Image
from targetedfool import targetedfool





net = models.resnet34(pretrained=True)
#net = models.vgg16(pretrained=True)
#net = models.vgg19(pretrained=True)
#net = models.densenet121(pretrained=True)
#net = models.inception_v3(pretrained=True)
#net = models.resnet152(pretrained=True)

net.eval()

im_orig = Image.open('test_im.jpg')

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

# Remove the mean
im = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])(im_orig)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)

tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                        transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224)])

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

r, loop_i, label_orig, label_pert, pert_image = target_deepfool(im, net,target=9)

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

tf(im).save('img/original.jpg')
tf(pert_image.cpu()[0]).save('img/adversarial.jpg')
tensor_to_PIL(pert_image-im).save('img/perturbation.jpg')

