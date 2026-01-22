import os
import shutil
import random

def split_and_copy_npy(input_path="../../data/dataset/all/", output_path="../../data/dataset/"):
    files = [f for f in os.listdir(input_path) if f.endswith('.npy')]
    random.shuffle(files)

    n = len(files)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split_name, split_files in splits.items():
        dst_dir = os.path.join(output_path, split_name)
        os.makedirs(dst_dir, exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(input_path, f), os.path.join(dst_dir, f))

if __name__ == "__main__":
    split_and_copy_npy()
