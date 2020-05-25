import xarray
import pandas as pd
import seaborn
import os
import pickle

from neural_structural_optimization import pipeline_utils
from neural_structural_optimization import problems
from neural_structural_optimization import models
from neural_structural_optimization import topo_api
from neural_structural_optimization import train_switch
import matplotlib.pyplot as plt


def train_all(example,switch,name,max_iterations,cnn_layers,cnn_kwargs):

    args = topo_api.specified_task(example)
    if cnn_kwargs is None:
        cnn_kwargs = {}

    #CNN MODEL - SWITCH
    model = models.CNNModel(args=args, **cnn_kwargs, conv_filters= cnn_layers)

    if os.path.exists("results/cnn_{}.pkl".format(name+"_"+str(switch))):
        print("Loading from file")
    else:
        ds_cnn=train_switch.train_lbfgs(model, switch)
        print(ds_cnn)
        z = model(None)
        pickle.dump((z,ds_cnn), open("results/cnn_{}.pkl".format(name+"_"+str(switch)), 'wb'))
    z, ds_cnn = pickle.load(open("results/cnn_{}.pkl".format(name+"_"+str(switch)), 'rb'))

    z_cnn =ds_cnn.design.sel(step=switch)
    z_cnn = z_cnn.to_masked_array().data.reshape(1, z_cnn.shape[0], z_cnn.shape[1])
    plt.imshow(z_cnn[0])
    plt.savefig("results/"+name+"_switch_"+str(switch)+"_cnn_switch"+".pdf")
    plt.close()
    loss_cnn_switch = float(ds_cnn.loss[switch])

    #PIXEL MODEL- INITIALIZED WITH CNN SWITCH MODEL

    pixel_model = models.PixelModel(args=args)
    if os.path.exists("results/pixel_switch_{}.pkl".format(name+"_"+str(switch))):
        print("Loading from file")
    else:
        ds_pixel_switch = train_switch.train_lbfgs(pixel_model, max_iterations-switch, init_model=lambda a: z)
        pickle.dump(ds_pixel_switch, open("results/pixel_switch_{}.pkl".format(name+"_"+str(switch)), 'wb'))
    ds_pixel_switch = pickle.load(open("results/pixel_switch_{}.pkl".format(name+"_"+str(switch)), 'rb'))

    step_pixel_switch =(len(ds_pixel_switch.design.loc[:,0].step)-1)
    z_pixel_switch =ds_pixel_switch.design.sel(step=step_pixel_switch)
    z_pixel_switch = z_pixel_switch.to_masked_array().data.reshape(1, z_pixel_switch.shape[0], z_pixel_switch.shape[1])
    plt.imshow(z_pixel_switch[0])
    plt.savefig("results/"+name+"_switch_"+str(switch)+"_pixel_switch"+".pdf")
    plt.close()
    loss_switch = float(ds_pixel_switch.loss.loc[step_pixel_switch])

    #PIXEL MODEL WITH NO INITIALIZATION

    pixel_model = models.PixelModel(args=args)
    if os.path.exists("results/pixel_{}.pkl".format(name)):
        print("Loading results")
    else:
        ds_pixel = train_switch.train_lbfgs(pixel_model, max_iterations, init_model=None)
        pickle.dump(ds_pixel, open("results/pixel_{}.pkl".format(name), 'wb'))
        step_pixel =(len(ds_pixel.design.loc[:,0].step)-1)
        z_pixel = ds_pixel.design.sel(step=step_pixel)
        z_pixel = z_pixel.to_masked_array().data.reshape(1, z_pixel.shape[0], z_pixel.shape[1])
        plt.imshow(z_pixel[0])
        plt.savefig("results/"+name+"_pixel_no_switch"+".pdf")
        plt.close()
    ds_pixel = pickle.load(open("results/pixel_{}.pkl".format(name), 'rb'))
    step_pixel =(len(ds_pixel.design.loc[:,0].step)-1)
    loss_no_switch = float(ds_pixel.loss.loc[step_pixel])

    #CNN MODEL- MAX ITERATIONS

    model = models.CNNModel(args=args, **cnn_kwargs, conv_filters= cnn_layers)

    if os.path.exists("results/cnn_{}.pkl".format(name+"_"+str(max_iterations))):
        print("Loading results")
    else:
        ds_cnn_all=train_switch.train_lbfgs(model,max_iterations)
        z = model(None)
        pickle.dump((ds_cnn_all), open("results/cnn_{}.pkl".format(name+"_"+str(max_iterations)), 'wb'))
    ds_cnn_all = pickle.load(open("results/cnn_{}.pkl".format(name+"_"+str(max_iterations)), 'rb'))

    z_cnn_all =ds_cnn_all.design.sel(step=max_iterations)
    z_cnn_all = z_cnn_all.to_masked_array().data.reshape(1, z_cnn_all.shape[0], z_cnn_all.shape[1])
    plt.imshow(z_cnn_all[0])
    plt.savefig("results/"+name+"_switch_"+str(switch)+"_CNN_max_iterations"+".pdf")
    plt.close()
    loss_cnn_all = float(ds_cnn_all.loss[max_iterations])

    losses = [loss_cnn_switch, loss_cnn_all, loss_switch,loss_no_switch]
    print(losses)

    print(ds_cnn_all.loss.to_pandas().cummin().plot(linewidth=2, label="cnn"))
    print(ds_pixel_switch.loss.to_pandas().cummin().loc[:max_iterations-switch-1].plot(linewidth=2, label="pixel_switch"))
    print(ds_pixel.loss.to_pandas().cummin().loc[:max_iterations-switch-1].plot(linewidth=2, label="pixel_no_switch"))

    plt.legend()
    plt.yscale("log")
    plt.ylabel('Compliance (loss)')
    plt.xlabel('Optimization step')
    plt.savefig("results/"+name+"_switch_"+str(switch)+"_losses"+".pdf")
    plt.close()
    seaborn.despine()

    return losses
