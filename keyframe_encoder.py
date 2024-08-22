from PIL import Image
from tqdm import tqdm
import clip
import numpy as np
import os
import torch

import h5py

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

def create_feature_vector(image_dir_path):
    h5py_file = image_dir_path.split('/')[-1]
    print(h5py_file)
    images = []
    image_paths = []
    img_dir_path = os.path.join(image_dir_path, "keyframes")
    for video in sorted(os.listdir(img_dir_path)):
        print(video)
        if not os.path.isdir(os.path.join(img_dir_path, video)):
            continue
        for img_file in tqdm(os.listdir(os.path.join(img_dir_path, video))):
            if not img_file.endswith(".jpg"):
                continue
            image = Image.open(os.path.join(img_dir_path, video, img_file)).convert("RGB")
            images.append(preprocess(image))
            image_paths.append(os.path.join(img_dir_path, video, img_file))
    image_input = torch.tensor(np.stack(images)).to(device)
    print(image_input.shape)
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().numpy()
    new_path = []
    for i in image_paths:
        new_path.append(i.replace('\\', '/'))
    new_path_encoded = np.array(new_path, dtype='S')
    
    with h5py.File(f"{h5py_file}.hdf5", "w") as data_file:
        data_file.create_dataset("data", data=image_features)
        data_file.create_dataset("ids", data=new_path_encoded)
    
def main():
    key_framelist = ["keyframe/Keyframes_L06"]
    for i in key_framelist:
        create_feature_vector(i)

if __name__ == "__main__":
    main()