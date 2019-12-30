#coding:utf-8
import pickle as p
import numpy as np
from PIL import Image


def load_CIFAR_batch(filename):
    with open(filename, 'rb')as f:
        datadict = p.load(f)
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


item = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

if __name__ == "__main__":
    imgX, imgY = load_CIFAR_batch("./Cifar-10/cifar-10-batches-py/data_batch_2")
    print (imgX.shape)

    di = {v: k for k, v in item.items()}

    for i in range(imgX.shape[0]):
        imgs = imgX[i - 1]
        if i < 200:
            img0 = imgs[0]
            img1 = imgs[1]
            img2 = imgs[2]
            i0 = Image.fromarray(img0)
            i1 = Image.fromarray(img1)
            i2 = Image.fromarray(img2)
            img = Image.merge("RGB", (i0, i1, i2))

            pred = di[imgY[i - 1]]
            name = "img" + str(i) + "_" + str(pred) + ".png"
            img.save("./images/" + pred + "/" + name, "png")

    print ("save.")
