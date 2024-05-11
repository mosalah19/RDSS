from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, resnet18, vgg16_bn, VGG16_BN_Weights, resnet101, mobilenet_v3_small, MobileNet_V3_Small_Weights, inception_v3, Inception_V3_Weights
from torch import nn
import torch.nn as nn
import os
import shutil
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train_model import *

Source_train = r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\data\archiveG\eyepac-light-v2-512-jpg\train'
Source_val = r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\data\archiveG\eyepac-light-v2-512-jpg\validation'
Source_test = r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\data\archiveG\eyepac-light-v2-512-jpg\test'


def make_folder(Source, csv_name: str):
    list_classe = os.listdir(Source)
    df_a = pd.DataFrame(columns=['Fundus', 'diag'])

    for classe in list_classe:
        path = os.path.join(Source, classe)
        list_images = [f for f in os.listdir(
            path) if os.path.isfile(os.path.join(path, f))]
        print(f'{classe} = {len(list_images)}')
        for image_name in list_images:
            df_a.loc[len(df_a.index)] = [image_name, classe]

    df_a = df_a.replace({'RG': 1, 'NRG': 0})
    n = len(df_a)
    stratified = df_a.groupby('diag', group_keys=False)\
        .apply(lambda x: x.sample(int(np.rint(n*len(x)/len(df_a)))))\
        .sample(frac=1).reset_index(drop=True).to_csv(f'{csv_name}.csv', index=False)


# make_folder(Source_train, "G_train")
# make_folder(Source_val, "G_val")
# make_folder(Source_test, "G_test")


model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
# for param in model.features.parameters():
#     param.requires_grad = False

model.classifier[-1] = nn.Linear(in_features=1024, out_features=1)
model.load_state_dict(torch.load(
    'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\model_for_G.pth'))


class load_dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()
        self.annotaions = pd.read_csv(csv_file)
        self.annotaions.replace
        self.root_dir = root_dir
        self.Transform = transform

    def __len__(self):
        return len(self.annotaions)

    def __getitem__(self, index):
        label_name = "Normal" if self.annotaions.iloc[index,
                                                      1] == 0 else "glaucoma"
        image_path = os.path.join(
            self.root_dir, label_name, self.annotaions.iloc[index, 0])
        image = plt.imread(image_path)
        # image = (image - image.min())/image.max()
        y_label = torch.tensor(self.annotaions.iloc[index, 1])
        if self.Transform:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            image = self.Transform(image)
        return [image, y_label]


# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to match MobileNetV3 input size
#     transforms.ToTensor(),           # Convert image to tensor
#     transforms.Normalize(            # Normalize image
#         mean=[0.485, 0.456, 0.406],  # Mean values for RGB channels
#         # Standard deviation values for RGB channels
#         std=[0.229, 0.224, 0.225]
#     )
# ])

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),       # Convert image to tensor
    transforms.Normalize(            # Normalize image
        mean=[0.485, 0.456, 0.406],  # Mean values for RGB channels
        # Standard deviation values for RGB channels
        std=[0.229, 0.224, 0.225]
    )
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
])


training = load_dataset(root_dir=Source_train,
                        csv_file=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\G_train.csv', transform=train_transform)
val = load_dataset(root_dir=Source_val,
                   csv_file=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\G_val.csv', transform=train_transform)
test = load_dataset(root_dir=Source_test,
                    csv_file=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\G_test.csv', transform=train_transform)
test_out = load_dataset(root_dir=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\data\ODIR_after_processed\Organization_test',
                        csv_file=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\G_N_out.csv', transform=train_transform)
training_data = DataLoader(dataset=training, batch_size=8, shuffle=True)
val_data = DataLoader(dataset=val, batch_size=8, shuffle=False)
test_data = DataLoader(dataset=test, batch_size=8, shuffle=False)
test_out_data = DataLoader(dataset=test_out, batch_size=8, shuffle=False)


# print(next(iter(training_data))[1].shape)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()
reduce_lr = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.85, patience=2, min_lr=1e-5)
train(model=model,
      train_dataloader=training_data,
      test_dataloader=test_out_data,
      loss_fn=loss_fn,
      save_name="Galucoma",
      optimizer=optimizer,
      epochs=10,
      print_freq=100)
