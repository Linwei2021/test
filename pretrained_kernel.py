import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.signal
import torch
import numpy as np

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def batch(img,mask, l, bs, nc, device):
    data = torch.empty([bs, nc, l,l]).to(device)
    mask_sample = torch.empty([bs, nc, l,l]).to(device)
    x_max, y_max = img.shape[:]
    for i in range(bs):
        # n = np.random.randint(0,n_max)
        img_slice = img
        mask_slice = mask
        x = np.random.randint(0, x_max - l)
        y = np.random.randint(0, y_max-l)
        data[i,:,:,:] = img_slice[x:x + l, y:y+l]
        mask_sample[i, :, :, :] = torch.from_numpy(mask_slice[x:x + l, y:y + l])

    return data,mask_sample

def sobel_filter(image):
    # h = image.shape[0]
    # w = image.shape[1]
    # image_new = np.zeros(image.shape, np.uint8)
    #
    # for i in range(1, h - 1):
    #     for j in range(1, w - 1):
    #         sx = (image[i + 1][j - 1] + 2 * image[i + 1][j] + image[i + 1][j + 1]) - \
    #              (image[i - 1][j - 1] + 2 * image[i - 1][j] + image[i - 1][j + 1])
    #         sy = (image[i - 1][j + 1] + 2 * image[i][j + 1] + image[i + 1][j + 1]) - \
    #              (image[i - 1][j - 1] + 2 * image[i][j - 1] + image[i + 1][j - 1])
    #         image_new[i][j] = np.sqrt(np.square(sx) + np.square(sy))

    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)

    image_new = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


    return image_new


def laplacian_filter(image):
    # h = image.shape[0]
    # w = image.shape[1]
    # image_new = np.zeros(image.shape, np.uint8)
    # for i in range(1, h-1):
    #     for j in range(1, w-1):
    #         image_new[i][j] = image[i + 1][j] + image[i - 1][j] + image[i][j + 1] + image[i][j - 1] - 8 * image[i][j]
    gray_lap = cv2.Laplacian(image, cv2.CV_16S, ksize=5)
    image_new = cv2.convertScaleAbs(gray_lap)
    return image_new


def mean(image):
    img_mean = cv2.blur(image, (5, 5))
    return img_mean

def gaussian(image):
    img_Guassian = cv2.GaussianBlur(image, (5, 5), 0)
    return img_Guassian

def median(image):
    img_median = cv2.medianBlur(image, 5)
    return img_median

def filter(batch,device):
    filtered = torch.empty([batch.shape[0], 5, batch.shape[2],batch.shape[3]]).to(device)#10,5,64,64

    for i in range (0,batch.shape[0]):
        filtered[i,0,:,:] = torch.from_numpy(sobel_filter(batch[i,0,:,:].numpy()))
        filtered[i, 1, :, :] = torch.from_numpy(laplacian_filter(batch[i, 0, :, :].numpy()))
        filtered[i, 2, :, :] = torch.from_numpy(mean(batch[i, 0, :, :].numpy()))
        filtered[i, 3, :, :] = torch.from_numpy(gaussian(batch[i, 0, :, :].numpy()))
        filtered[i, 4, :, :] = torch.from_numpy(median(batch[i, 0, :, :].numpy()))
    return filtered


if __name__ == "__main__":
    tif = torch.Tensor(tifffile.imread('data/feature-stack0001_test.tif')[0]).to(device)

    mask = tifffile.imread('data/mask.tif')
    dataset, mask1 = batch(tif, mask, 64, 10, 1, device)#10,1,64,64
    # filtered = filter(tif,device)
    # plt.imshow(filtered.detach()[0][0].cpu(), cmap='gray')
    plt.imshow(laplacian_filter(tif.numpy()),cmap='gray')
    plt.show()



