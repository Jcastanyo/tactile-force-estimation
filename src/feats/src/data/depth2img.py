import os
import glob

import cv2
import numpy as np

from depth_gs_mini import gsdevice, gs3drecon


def load_data(path):

    npy_files = glob.glob(os.path.join(path, "*.npy"))
    return npy_files

def load_images(path):
    img_files = glob.glob(os.path.join(path, "*.png"))
    return img_files


def img_to_depth(img):
    pass


def main():

    path = "../../data/dataset/final_data/"
    zeroing_img_path = "./depth_gs_mini/zero_depth_data/"

    output_path = "/home/aurova/Desktop/julio/remain_project/tactile-force-estimation/src/feats/data/dataset/final_data/train_dm/"
    
    #train_data = load_data(os.path.join(path, "train"))
    #val_data = load_data(os.path.join(path, "val"))
    #test_data = load_data(os.path.join(path, "test"))

    train_data = load_images(os.path.join(path, "train_no_mark"))
    val_data = load_images(os.path.join(path, "val_no_mark"))
    test_data = load_images(os.path.join(path, "test_no_mark"))

    zeroing_img_files = load_images(zeroing_img_path)

    dev = gsdevice.Camera("gs_mini")
    net_path = os.path.join("./depth_gs_mini/nnmini.pt")
    nn = gs3drecon.Reconstruction3D(dev)
    net = nn.load_nn(net_path, "cuda")

    min_value = 0.0
    max_value = 0.0026#0.0022
    mmpp = 0.0634  # mini gel 18x24mm at 240x320
    mpp = mmpp / 1000.
    height, width = dev.imgh, dev.imgw

    #img_bk = cv2.imread("./nocontact.png")
    #img_bk = cv2.cvtColor(img_bk, cv2.COLOR_BGR2RGB)
    #img_bk = cv2.resize(img_bk, (240, 320))
    
    #for i in range(100):
    #    dm = nn.get_depthmap(img_bk, False)

    for i in range(len(zeroing_img_files)):
        img_bk = cv2.imread(zeroing_img_files[i])
        #img_bk = cv2.cvtColor(img_bk, cv2.COLOR_BGR2RGB)
        img_bk = cv2.resize(img_bk, (240, 320))
        dm = nn.get_depthmap(img_bk, False)
        


    for i in range(len(train_data)):
        #data = np.load(train_data[i], allow_pickle=True).item()
        #img_no_mark = data["gs_img_no_mark"]
        img_no_mark = cv2.imread(train_data[i])
        #img_no_mark = cv2.cvtColor(img_no_mark, cv2.COLOR_BGR2RGB)
        img_no_mark = cv2.resize(img_no_mark, (240, 320))
    
        dm = nn.get_depthmap(img_no_mark, False) * mpp
        dm -= np.min(dm)

        img_dm = (255.0 * ((dm - min_value) / ((max_value-np.min(dm)) - min_value)))
        img_dm = img_dm.astype(np.uint8)

        img_no_mark = cv2.resize(img_no_mark, (320, 240))
        img_dm = cv2.resize(img_dm, (320, 240))

        cv2.imwrite(output_path + train_data[i].split("/")[-1], img_dm)

        #cv2.imshow("rgb", img_no_mark)
        #cv2.imshow("dm", img_dm)

        #if cv2.waitKey(0) & 0xFF == ord('q'):
        #    continue
        #else:
        #    break
        
        

    


if __name__ == "__main__":
    main()
