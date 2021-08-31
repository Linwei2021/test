from semi_supervised.unet import Net
# from semi_supervised.network import Net
from semi_supervised.util import *
from semi_supervised.pretrained_kernel import *
import torch.optim as optim
import tifffile
import matplotlib.pyplot as plt
import matplotlib
import wandb
# matplotlib.use("Agg")
wandb.init(project='semi-supervised', entity='linwei2021')
wandb.run.name = "1000-100labeled-nmc-unet"
config = wandb.config
config.learning_rate = 1e-3
config.momentum = 0.9
config.l = 64
config.bs = 10
config.nc = 1
config.num_epochs = 1000

config.num = 10 #number of labeled pixel
config.pixel = 10
config.weight = 1
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    #NMC
    tif = torch.Tensor(tifffile.imread('data/feature-stack0001_test.tif')[0]).to(device)
    mask = tifffile.imread('data/mask.tif')

    # SOFC
    # tif = torch.Tensor(tifffile.imread('data/test.tif')).to(device)
    # tif = tif / 255
    #mask = tifffile.imread('data/cathode_segmented_tiff_z001.tif')

    loss_function = MylossFunc()

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    num_pore = 0
    num_am = 0
    num_cbd = 0

    labled, mask2 = btach(tif,mask,config.l,1,config.nc,device)#1,1,64,64

    wandb.watch(model)
    for num in range(0,config.num_epochs ):
        dataset, mask1 = batch(tif, mask, config.l, config.bs, config.nc, device)  # 10,1,64,64

        filtered = filter(dataset, device)
        trained = torch.cat([dataset, filtered], 1).to(device)#10,6,64,64

        optimizer.zero_grad()

        # test
        dataset_test = dataset[0:1, :, :, :].to(device)  # 1,1,64,64

        output = model(dataset)  # 10,3,64,64

        labeled_x = torch.zeros(config.num).to(device)
        labeled_y = torch.zeros(config.num).to(device)
        labeled_z = torch.zeros(config.num).to(device)
        index = []

        if num < config.pixel:
            labeled_x, labeled_y, labeled_z, index, num_pore, num_am, num_cbd = get_pixel(config.num, mask1, num_pore,
                                                                                          num_am, num_cbd, device)
            wandb.log({"labeled pore": num_pore, "labeled active material": num_am, "labeled cbd": num_cbd})

        loss = loss_function(output, labeled_x, labeled_y, labeled_z, index, config.weight,device)

        loss = loss.mean()
        loss.backward()
        optimizer.step()
        wandb.log({"Total loss": loss})
        if num % 50 == 0:
            print(num)
            print(loss.item())
            fig = plt.figure()
            fig.add_subplot(221)
            plt.imshow(dataset_test[0][0].cpu(), cmap='gray')
            fig.add_subplot(222)
            plt.imshow(output.detach()[0][0].cpu(), cmap='gray')
            fig.add_subplot(223)
            plt.imshow(output.detach()[0][1].cpu(), cmap='gray')
            fig.add_subplot(224)
            plt.imshow(output.detach()[0][2].cpu(), cmap='gray')
            plt.savefig('semi_supervised_output/output_' + str(num) + '.png')
            wandb.log({"Segmentation output (train sample)": wandb.Image(fig)})
            plt.close('all')

            output = output.detach().cpu().numpy()
            mask1 = mask1.cpu().numpy()
            plotted = plot(output, mask1)
            # plotted.show()
            wandb.log({"Plotted pixel (train sample)": wandb.Image(plotted)})
    #             add plot of triangle coordinates here too

    torch.save(model.state_dict(), 'FNN.pth')

def test():
    loss_function = MylossFunc()

    #NMC
    tif = torch.Tensor(tifffile.imread('data/feature-stack0001_test.tif')[0]).to(device)
    test_img = tif[np.newaxis, np.newaxis, :, :]
    mask = tifffile.imread('data/mask.tif')
    mask_img = mask[np.newaxis, np.newaxis, :, :]

    # SOFC
    # tif = torch.Tensor(tifffile.imread('data/test.tif')).to(device)
    # tif = tif / 255
    # test_img = tif[np.newaxis, np.newaxis, :, :]
    # mask = tifffile.imread('data/cathode_segmented_tiff_z001.tif')
    # mask_img = mask[np.newaxis, np.newaxis, :, :]

    model = Net().to(device)
    model.load_state_dict(torch.load('FNN.pth'))

    model.eval()

    filtered = filter(test_img, device)
    tested = torch.cat([test_img, filtered], 1).to(device)  # 10,6,64,64

    result = model(test_img)
    # loss = loss_function.forward(result)
    # loss = loss.mean()
    # print("test loss", loss.item())

    fig = plt.figure()

    fig.add_subplot(221)
    plt.imshow(result.detach()[0][0].cpu(), cmap='gray')
    fig.add_subplot(222)
    plt.imshow(result.detach()[0][1].cpu(), cmap='gray')
    fig.add_subplot(223)
    plt.imshow(result.detach()[0][2].cpu(), cmap='gray')
    plt.savefig('semi_supervised'
                '_results/segmented.png')
    wandb.log({"Segmentation output (input image)": wandb.Image(fig)})
    plt.close('all')

    result = result.detach().cpu().numpy()
    plotted = plot(result, mask_img)
    phasea, phaseb, phasec = one_hot(result)

    phasea = np.array(phasea)
    phaseb = np.array(phaseb)
    phasec = np.array(phasec)
    np.savetxt(fname="semi_supervised_results/dataA.csv", X=phasea, fmt="%d", delimiter=",")
    np.savetxt(fname="semi_supervised_results/dataB.csv", X=phaseb, fmt="%d", delimiter=",")
    np.savetxt(fname="semi_supervised_results/dataC.csv", X=phasec, fmt="%d", delimiter=",")

    wandb.log({"Plotted pixel (output)": wandb.Image(plotted)})
    plotted.savefig('semi_supervised_results/plotted.png')

    phase1, phase2, phase3 = plot_mask(mask)
    phase1 = np.array(phase1)
    phase2 = np.array(phase2)
    phase3 = np.array(phase3)
    np.savetxt(fname="semi_supervised_results/data1.csv", X=phase1, fmt="%d", delimiter=",")
    np.savetxt(fname="semi_supervised_results/data2.csv", X=phase2, fmt="%d", delimiter=",")
    np.savetxt(fname="semi_supervised_results/data3.csv", X=phase3, fmt="%d", delimiter=",")

    ##Compare groundtruth and results##
    # print(phase1)
    accuracy = 0
    error1 = 0
    error2 = 0
    error3 = 0
    index_pore = []
    index_material = []
    index_cbd = []

    print(phasea.flatten().tolist().count(1))
    print(phaseb.flatten().tolist().count(1))
    print(phasec.flatten().tolist().count(1))
    for i in range(phase1.shape[0]):
        for l in range(phase1.shape[1]):
            if phase1[i][l] == 1:
                index_pore.append(i)
                index_pore.append(l)
                if phasea[i][l] == 1:
                    accuracy += 1
                else:
                    error1 += 1

            if phase2[i][l] == 1:
                index_material.append(i)
                index_material.append(l)
                if phaseb[i][l] == 1:
                    accuracy += 1
                else:
                    error2 += 1
            if phase3[i][l] == 1:
                index_cbd.append(i)
                index_cbd.append(l)
                if phasec[i][l] == 1:
                    accuracy += 1
                else:
                    error3 += 1
    accuracy = accuracy / (phase1.shape[0] * phase1.shape[1])

    print("accuracy is ", accuracy)

    error1 = error1 / phase1.flatten().tolist().count(1)
    error2 = error2 / phase2.flatten().tolist().count(1)
    error3 = error3 / phase3.flatten().tolist().count(1)

    print("Error rate for pore ", error1 * 100, "%")
    print("Error rate for active material ", error2 * 100, "%")
    print("Error rate for CBD ", error3 * 100, "%")

    pore_x = []
    pore_y = []

    for i in range(0,len(index_pore)-1,2):

        if phasea[index_pore[i]][index_pore[i+1]]==1:
            pore_x.append(0)
            pore_y.append(0)

        if phaseb[index_pore[i]][index_pore[i+1]]==1:
            pore_x.append(1)
            pore_y.append(0)

        if phasec[index_pore[i]][index_pore[i+1]]==1:
            pore_x.append(0.5)
            pore_y.append(0.5*np.sqrt(3))

    # print(len(pore_x))
    # print(pore_x)
    fig = plt.figure()
    fig.add_subplot(221)
    plt.scatter(pore_x, pore_y)
    plt.xlim((0, 1))
    fig.add_subplot(222)
    plt.hist2d(pore_x, pore_y, norm=LogNorm())
    plt.xlim((0, 1))
    plt.colorbar()
    plt.savefig('results/pore distribution.png')
    wandb.log({"pore distribution": wandb.Image(fig)})
    plt.close('all')

    material_x = []
    material_y = []

    for i in range(0,len(index_material)-1,2):

        if phasea[index_material[i]][index_material[i+1]] == 1:
            material_x.append(0)
            material_y.append(0)

        if phaseb[index_material[i]][index_material[i+1]] == 1:
            material_x.append(1)
            material_y.append(0)

        if phasec[index_material[i]][index_material[i+1]] == 1:
            material_x.append(0.5)
            material_y.append(0.5 * np.sqrt(3))


    fig = plt.figure()
    fig.add_subplot(221)
    plt.scatter(material_x, material_y)
    plt.xlim((0, 1))
    fig.add_subplot(222)
    plt.hist2d(material_x, material_y, norm=LogNorm())
    plt.xlim((0, 1))
    plt.colorbar()
    plt.savefig('results/active material distribution.png')
    wandb.log({"active material distribution": wandb.Image(fig)})
    plt.close('all')

    cbd_x = []
    cbd_y = []

    for i in range(0,len(index_cbd)-1,2):

        if phasea[index_cbd[i]][index_cbd[i+1]] == 1:
            cbd_x.append(0)
            cbd_y.append(0)

        if phaseb[index_cbd[i]][index_cbd[i+1]] == 1:
            cbd_x.append(1)
            cbd_y.append(0)

        if phasec[index_cbd[i]][index_cbd[i+1]] == 1:
            cbd_x.append(0.5)
            cbd_y.append(0.5 * np.sqrt(3))

    fig = plt.figure()
    fig.add_subplot(221)
    plt.scatter(cbd_x, cbd_y)
    plt.xlim((0, 1))
    fig.add_subplot(222)
    plt.hist2d(cbd_x, cbd_y,  norm=LogNorm())
    plt.xlim((0, 1))
    plt.colorbar()
    plt.savefig('results/CBD distribution.png')
    wandb.log({"CBD distribution": wandb.Image(fig)})
    plt.close('all')
