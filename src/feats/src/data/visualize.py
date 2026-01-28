import os.path
import sys; sys.path.append("../")
import argparse
import yaml
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from data.dataloader import FEATSDataset, normalize, unnormalize
from models.unet import UNet


def plot_img(img, loc, w_name):
    """
    Plot the image.

    :param img: image to plot
    :param loc: location of the window
    :return: None
    """

    img = cv2.cvtColor(img.numpy().squeeze(0), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 240))
    cv2.imshow(w_name, cv2.resize(img, (int(320*1.5), int(240*1.5))))
    cv2.moveWindow(w_name, loc[0], loc[1])



def plot_force_map_lbl(data):
    """
    Plot the force map.

    :param data: dictionary containing the data
    :return: None
    """
    
    # unnormalize gt data
    gt_grid_x = unnormalize(data["grid"][:, :, :, 0].squeeze(0), "grid_x", config["norm_file"])
    gt_grid_y = unnormalize(data["grid"][:, :, :, 1].squeeze(0), "grid_y", config["norm_file"])
    gt_grid_z = unnormalize(data["grid"][:, :, :, 2].squeeze(0), "grid_z", config["norm_file"])
    gt_total_force_x = unnormalize(data["total_force"][:, 0].squeeze(0), "f_x", config["norm_file"])
    gt_total_force_y = unnormalize(data["total_force"][:, 1].squeeze(0), "f_y", config["norm_file"])
    gt_total_force_z = unnormalize(data["total_force"][:, 2].squeeze(0), "f_z", config["norm_file"])

    # create figure
    fig = plt.figure()
    fig.set_size_inches(12, 7)

    # create grid for 3x2 subplots
    grid = AxesGrid(fig, 121,
            nrows_ncols=(2, 3),
            axes_pad=0.50,
            share_all=True,
            label_mode="all",
            cbar_location="bottom",
            cbar_mode="edge",
            cbar_pad=0.25,
            cbar_size="15%",
            direction="column"
            )

    # set color limits
    clim_xy = [-0.07, 0.07]
    clim_z = [-0.3, 0.0]

    # plot each channel
    gt_x = grid[0].imshow(gt_grid_x, cmap="viridis")
    gt_x.set_clim(clim_xy)
    #pred_x = grid[1].imshow(pred_grid_x, cmap="viridis")
    #pred_x.set_clim(clim_xy)
    grid.cbar_axes[0].colorbar(gt_x)

    gt_y = grid[2].imshow(gt_grid_y, cmap="viridis")
    gt_y.set_clim(clim_xy)
    #pred_y = grid[3].imshow(pred_grid_y, cmap="viridis")
    #pred_y.set_clim(clim_xy)
    grid.cbar_axes[1].colorbar(gt_y)

    gt_z = grid[4].imshow(gt_grid_z, cmap="viridis")
    gt_z.set_clim(clim_z)
    #pred_z = grid[5].imshow(pred_grid_z, cmap="viridis")
    #pred_z.set_clim(clim_z)
    grid.cbar_axes[2].colorbar(gt_z)

    # set labels for each axis
    j = 0
    direction = ["x", "y", "z"]
    for cax in grid.cbar_axes:
        cax.axis[cax.orientation].set_label("{}-axis [N]".format(direction[j%3]))
        j += 1

    # adjust the size of the plot
    plt.subplots_adjust(left=-.52, right=4)

    # use total force as title for each channel
    grid[0].set_title("{:.2f}N".format(gt_total_force_x))
    #grid[1].set_title("{:.2f}N".format(torch.sum(pred_grid_x, dim=[0, 1])))
    grid[2].set_title("{:.2f}N".format(gt_total_force_y))
    #grid[3].set_title("{:.2f}N".format(torch.sum(pred_grid_y, dim=[0, 1])))
    grid[4].set_title("{:.2f}N".format(gt_total_force_z))
    #grid[5].set_title("{:.2f}N".format(torch.sum(pred_grid_z, dim=[0, 1])))

    # set title for the rows
    grid[0].set_ylabel("Ground Truth", fontsize=12)
    grid[1].set_ylabel("Prediction", fontsize=12)



def main(config):
    """
    Main function for visualization of the UNet. This function loads the model,
    creates a test dataset and iterates over the dataset to get the predictions
    of the model. The predictions are then plotted against the ground truth.

    :param config: dictionary containing the configuration for visualization
    :return: None
    """

    # create dataset
    train_dataset = FEATSDataset(config["train_data"], config["norm_file"], normalize, train=False)
    val_dataset = FEATSDataset(config["val_data"], config["norm_file"], normalize, train=False)
    test_dataset = FEATSDataset(config["test_data"], config["norm_file"], normalize, train=False)

    # create data loader for test dataset
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # load calibration file
    if config["calibration_file"] is not None:
        calibration = np.load(config["calibration_file"])
        rows, cols = 240, 320
        M = np.float32([[1, 0, calibration[0]], [0, 1, calibration[1]]])

    for i, data in enumerate(test_loader):
            
        # prepare input data
        if config["calibration_file"] is not None:
            inputs_prewarp = data["gs_img"].numpy().squeeze(0)
            inputs_warp = cv2.warpAffine(inputs_prewarp,  M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
            inputs = torch.from_numpy(inputs_warp).permute(2, 0, 1).unsqueeze(0)
        else:
            inputs = data["gs_img"].permute(0, 3, 1, 2)

        # show image and make it bigger
        plot_img(data["gs_img"], (1320, 0), "GelSight Mini")
        plot_img(data["gs_img_no_mark_rec"], (1320, 400), "GM No Mark Rec")
        plot_img(data["dm_img"], (1320, 600), "DM")
        plot_img(data["gs_img_no_mark_rec_diff"], (1320, 800), "GM No Mark Rec Diff")

        plot_force_map_lbl(data)

        # show plot and close it after key press
        cv2.waitKey(100)
        plt.show()
        
    # close image
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Visualize UNet")
    parser.add_argument("-c", "--config", type=str, default="visualize_config.yaml", help="Path to config file for visualization")
    args = parser.parse_args()

    # load config file
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # visualize network predictions
    main(config)
