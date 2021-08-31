import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import math
import wandb
# matplotlib.use("Agg")
wandb.init(project='semi-supervised', entity='linwei2021')
wandb.run.name = "1000-100labeled-nmc-unet"


def batch(img, mask, l, bs, nc, device):
    data = torch.empty([bs, nc, l, l]).to(device)
    mask_sample = torch.empty([bs, nc, l, l]).to(device)
    x_max, y_max = img.shape[:]
    for i in range(bs):
        img_slice = img
        mask_slice = mask
        x = np.random.randint(0, x_max - l)
        y = np.random.randint(0, y_max - l)
        data[i, :, :, :] = img_slice[x:x + l, y:y + l]
        mask_sample[i, :, :, :] = torch.from_numpy(mask_slice[x:x + l, y:y + l])

    return data, mask_sample

def get_wholepixel(mask,num1,num2,num3,device):
    phase1 =[]
    phase2=[]
    phase3=[]

    for i in range(0,mask.shape[2]):
        for j in range(0,mask.shape[3]):
            if mask[:,:,i,j] == 0:
                phase1.append(i)
                phase1.append(j)
                num1+=1
            if mask[:,:,i,j] == 1:
                phase2.append(i)
                phase2.append(j)
                num2 += 1
            if mask[:,:,i,j] == 2:
                phase3.append(i)
                phase3.append(j)
                num3 +=1

    return phase1,phase2,phase3





def get_pixel(num, mask, num_pore, num_am,num_cbd, device):
    gt = mask[0:1, :, :, :].to(device)  # 1,1,64,64
    x_max = gt.shape[2]
    y_max = gt.shape[3]

    labeled_x = []
    labeled_y = []
    labeled_z = []
    index = []

    for i in range(0, num):

        x = np.random.randint(0, x_max)
        y = np.random.randint(0, y_max)
        index.append(x)
        index.append(y)

        if gt[:, :, x, y] == 0:
            labeled_x.append(1)
            labeled_y.append(0)
            labeled_z.append(0)
            num_pore+=1
        if gt[:, :, x, y] == 1:
            labeled_x.append(0)
            labeled_y.append(1)
            labeled_z.append(0)
            num_am+=1
        if gt[:, :, x, y] == 2:
            labeled_x.append(0)
            labeled_y.append(0)
            labeled_z.append(1)
            num_cbd+=1

    return labeled_x, labeled_y, labeled_z, index, num_pore,num_am,num_cbd

def plot(input,mask):
    pore_x = []
    pore_y = []
    pore_z = []

    material_x = []
    material_y = []
    material_z = []

    cbd_x = []
    cbd_y = []
    cbd_z = []

    x = []
    y = []
    z = []

    porex=[]
    materialx=[]
    cbdx=[]



    for i in range(mask.shape[2]):
        for l in range (mask.shape[3]):
            if mask[0, 0 ,i,l] == 0:
                porex.append(i)
                porex.append(l)
            if mask[0, 0 ,i,l] == 1:
                materialx.append(i)
                materialx.append(l)
            if mask[0, 0 ,i,l] == 2:
                cbdx.append(i)
                cbdx.append(l)


    ##Get pore
    for pixel in range(0,len(porex)-1,2):
        pore_x.append(input[0, 0, porex[pixel], porex[pixel+1]])
        pore_y.append(input[0, 1, porex[pixel], porex[pixel + 1]]-1)
        pore_z.append(input[0, 2, porex[pixel], porex[pixel + 1]])
    ##Get material
    for pixel in range(0,len(materialx)-1,2):
        material_x.append(input[0, 0, materialx[pixel], materialx[pixel+1]])
        material_y.append(input[0, 1, materialx[pixel], materialx[pixel + 1]]-1)
        material_z.append(input[0, 2, materialx[pixel], materialx[pixel + 1]])
    ##Get cbd
    for pixel in range(0,len(cbdx)-1,2):
        cbd_x.append(input[0, 0, cbdx[pixel], cbdx[pixel+1]])
        cbd_y.append(input[0, 1, cbdx[pixel], cbdx[pixel + 1]]-1)
        cbd_z.append(input[0, 2, cbdx[pixel], cbdx[pixel + 1]])


    ##Overall pixels
    for pixel in range(input.shape[2]):
        for pixel1 in range(input.shape[3]):
            x.append(input[0, 0, pixel, pixel1])
            y.append(input[0, 1, pixel, pixel1]-1)
            # z.append(0)
            z.append(input[0, 2, pixel, pixel1])

    x,y=convert(x,y,z)
    pore_x,pore_y = convert(pore_x,pore_y,pore_z)
    material_x, material_y = convert(material_x, material_y, material_z)
    cbd_x, cbd_y = convert(cbd_x, cbd_y, cbd_z)

    fig = plt.figure()
    fig.add_subplot(221)
    plt.scatter(x, y)
    plt.xlim((0,1))
    fig.add_subplot(222)
    plt.hist2d(x, y, bins=input.shape[2], norm=LogNorm())
    plt.xlim((0, 1))
    plt.colorbar()

    plt.close('all')

    fig1 = plt.figure()
    plt.scatter(pore_x,pore_y,s=input.shape[2]/10,alpha = 1/5)
    plt.scatter(material_x,material_y,s=input.shape[2]/10,alpha = 1/5)
    plt.scatter(cbd_x,cbd_y,s=input.shape[2]/10,alpha = 1/5)
    plt.xlim((0, 1))
    plt.legend(['active material','pore','CBD'],loc = 'upper left')
    wandb.log({"Plotted pixel with three phases":wandb.Image(fig1)})
    plt.close('all')

    fig2 = plt.figure()
    fig2.add_subplot(221)
    # plt.scatter(pore_x,pore_y,s=input.shape[2]/4,c='b')
    # plt.legend(['pore'], loc='upper left')
    plt.hist2d(pore_x,pore_y, bins = round(np.sqrt(len(pore_x))),norm=LogNorm())
    plt.xlim((0, 1))
    plt.colorbar()
    plt.title('active material')

    fig2.add_subplot(222)
    # plt.scatter(material_x,material_y,s=input.shape[2]/4,c='y')
    # plt.legend(['active material'], loc='upper left')
    plt.hist2d(material_x,material_y, bins = round(np.sqrt(len(material_x))),norm=LogNorm())
    plt.xlim((0, 1))
    plt.colorbar()
    plt.title('pore')

    fig2.add_subplot(223)
    # plt.scatter(cbd_x, cbd_y, s=input.shape[2] / 4,c='g')
    plt.legend(['CBD'], loc='upper left')
    plt.hist2d(cbd_x, cbd_y,bins=round(np.sqrt(len(cbd_x))), norm=LogNorm())
    plt.xlim((0, 1))
    plt.colorbar()
    plt.title('cbd')

    wandb.log({"Plotted pixel (three phases)": wandb.Image(fig2)})
    plt.close('all')


    ##3D
    # ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=10., azim=11)
    # ax.scatter(x1, y1, z1, cmap = 'jet_r')
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # fig.savefig('results/plotted with density.png')
    # wandb.log({"Plotted pixel with density": wandb.Image(fig)})
    # plt.show()
    return fig


def convert(x,y,z):

    # rotate 45 degree based on z
    x1 = []
    y1 = []
    z1 = []
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    theta = math.pi / 4

    num = x.size
    for i in range(num):
        x1.append(x[i] * math.cos(theta) - y[i] * math.sin(theta))
        y1.append(x[i] * math.sin(theta) + y[i] * math.cos(theta))
        z1.append(z[i])

#rotate to z = 0 based on x
    x2 = []
    y2 = []
    z2 = []
    x1 = np.array(x1)
    y1 = np.array(y1)
    z1 = np.array(z1)
    theta = math.atan(2/ math.sqrt(2))
    num = x1.size
    for i in range(num):
        x2.append(x1[i]/math.sqrt(2))
        y2.append(-(y1[i] * math.cos(theta) - z1[i] * math.sin(theta))/math.sqrt(2))
        z2.append(y1[i] * math.sin(theta) + z1[i] * math.cos(theta))

    return x2,y2

def one_hot(input):
    phase1 = np.zeros((input.shape[2], input.shape[3]), dtype=np.int)
    phase2 = np.zeros((input.shape[2], input.shape[3]), dtype=np.int)
    phase3 = np.zeros((input.shape[2], input.shape[3]), dtype=np.int)
    # print("shape",phase1.shape)

    for i in range(input.shape[2]):
        for l in range(input.shape[3]):
            result = np.array([input[0][0][i][l], input[0][1][i][l], input[0][2][i][l]])
            flag = np.argmax(result)
            if flag==0:
                phase1[i][l]=1
            if flag==1:
                phase2[i][l] = 1
            if flag == 2:
                phase3[i][l] = 1

    fig = plt.figure()
    fig.add_subplot(221)
    plt.imshow(phase1*255, cmap='gray')
    fig.add_subplot(222)
    plt.imshow(phase2*255, cmap='gray')
    fig.add_subplot(223)
    plt.imshow(phase3*255, cmap='gray')
    plt.savefig('results/one_hot_output.png')
    wandb.log({"One_hot_encoded_output":wandb.Image(fig)})
    plt.close('all')
    # plt.show()

    return phase1,phase2,phase3

def plot_mask(img):
    phase1 = np.zeros((img.shape[0],img.shape[1]),dtype = np.int)
    phase2 = np.zeros((img.shape[0],img.shape[1]),dtype = np.int)
    phase3= np.zeros((img.shape[0],img.shape[1]),dtype = np.int)


    for i in range(img.shape[0]):
        for l in range(img.shape[1]):
            if img[i][l] == 0:
                phase1[i][l] = 1
            if img[i][l] == 1:
                phase2[i][l] = 1
            if img[i][l] == 2:
                phase3[i][l] = 1


    # fig = plt.figure()
    # fig.add_subplot(221)
    # plt.imshow(phase1, cmap='gray')
    # fig.add_subplot(222)
    # plt.imshow(phase2, cmap='gray')
    # fig.add_subplot(223)
    # plt.imshow(phase3, cmap='gray')
    # plt.savefig('results/mask.png')
    # wandb.log({"Ground truth": wandb.Image(fig)})
    # plt.close('all')
    # plt.show()

    # fig1 = plt.figure()
    # ax = fig1.add_subplot(111, projection='3d')
    # ax.view_init(elev=10., azim=11)
    # ax.scatter(phase1.flatten(), phase2.flatten(), phase3.flatten())
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.savefig('results/mask_plotted.png')
    # wandb.log({"Ground truth plotted": wandb.Image(fig1)})

    return phase1,phase2,phase3





class MylossFunc(nn.Module):
    def __init__(self):
        super(MylossFunc, self).__init__()

    def forward(self, output, mask, predicted_x, predicted_y, predicted_z, index, weight, device):

        mean_square_loss = torch.tensor([0])
        l = 0

        for i in
        term1_1 = torch.pow(torch.sub(0-1,2))

        if len(index) > 0:

            for i in range(0, len(index), 2):
                term1_1 = torch.pow(torch.sub(predicted_x[l], output[0, 0, index[i], index[i + 1]]), 2).to(device)
                term1_2 = torch.pow(torch.sub(predicted_y[l], output[0, 1, index[i], index[i + 1]]), 2).to(device)
                term1_3 = torch.pow(torch.sub(predicted_z[l], output[0, 2, index[i], index[i + 1]]), 2).to(device)
                mean_square = torch.add(term1_1, term1_2).to(device)
                mean_square = torch.add(mean_square, term1_3).to(device)
                mean_square_loss = torch.add(mean_square, mean_square_loss).to(device)
                l = l + 1

            mean_square_loss = torch.div(mean_square_loss, l)
        wandb.log({"mean squared error loss": mean_square_loss})

        term1 = torch.multiply(torch.multiply((1 - output[:, 0]), output[:, 1]), output[:, 2]).to(device)
        term1 = torch.sum(term1, dim=[1, 2]).to(device)

        term2 = torch.multiply(torch.multiply((1 - output[:, 1]), output[:, 0]), output[:, 2]).to(device)
        term2 = torch.sum(term2, dim=[1, 2]).to(device)

        term3 = torch.multiply(torch.multiply((1 - output[:, 2]), output[:, 1]), output[:, 0]).to(device)
        term3 = torch.sum(term3, dim=[1, 2]).to(device)

        loss = torch.add(term1, term2).to(device)
        loss = torch.add(loss, term3).to(device)
        loss = torch.div(loss, output.shape[2] * output.shape[3]).to(device)
        # wandb.log({"unsupervised loss": loss})
        mean_square_loss = torch.multiply(mean_square_loss,weight)
        # loss = torch.multiply(loss,(1-weight))
        loss = torch.add(mean_square_loss, loss).to(device)

        # loss = Variable(loss, requires_grad=True)

        return loss
