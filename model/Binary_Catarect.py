from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet18, vgg16_bn, VGG16_BN_Weights, resnet101, ResNet18_Weights, inception_v3, Inception_V3_Weights
from torch import nn
import torch.nn as nn
import os
import shutil
import pandas as pd
import numpy as np
from train_model import *


def create_folder():
    # Source = r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\data\ODIR_after_processed\Organization_train'
    # list_classe = ['cataract', 'Normal']
    # df_a = pd.DataFrame(columns=['Fundus', 'diag'])
    # for classe in list_classe:
    #     path = os.path.join(Source, classe)
    #     list_images = [f for f in os.listdir(
    #         path) if os.path.isfile(os.path.join(path, f))]
    #     if classe == 'Normal':
    #         list_images[:300]
    #     print(f'{classe} = {len(list_images)}')
    #     for image_name in list_images:
    #         df_a.loc[len(df_a.index)] = [image_name, classe]

    # df_a = df_a.replace({'cataract': 1, 'Normal': 0})
    # n = len(df_a)
    # stratified = df_a.groupby('diag', group_keys=False)\
    #     .apply(lambda x: x.sample(int(np.rint(n*len(x)/len(df_a)))))\
    #     .sample(frac=1).reset_index(drop=True).to_csv('C_N.csv', index=False)

    Source = r'D:\Graduated Project\Week1_1\source_4\eyedata2'
    list_classe = ['cataract', 'Normal']
    df_a = pd.DataFrame(columns=['Fundus', 'diag'])

    for classe in list_classe:
        path = os.path.join(Source, classe)
        list_images = [f for f in os.listdir(
            path) if os.path.isfile(os.path.join(path, f))]
        print(f'{classe} = {len(list_images)}')
        for image_name in list_images:
            df_a.loc[len(df_a.index)] = [image_name, classe]

    df_a = df_a.replace({'cataract': 1, 'Normal': 0})
    n = len(df_a)
    stratified = df_a.groupby('diag', group_keys=False)\
        .apply(lambda x: x.sample(int(np.rint(n*len(x)/len(df_a)))))\
        .sample(frac=1).reset_index(drop=True).to_csv('C_N_test_out.csv', index=False)


create_folder()

model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(nn.Linear(25088, 64),
                                 nn.ReLU(),
                                 nn.Dropout(0.4),
                                 nn.Linear(64, 2)

                                 )

model.load_state_dict(torch.load(
    r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\Cataract.pth'))
training_image_path = Path(
    r'data/ODIR_after_processed/Organization_train')
test_image_path = Path(
    r'D:\Graduated Project\Week1_1\source_4\eyedata2')


class load_dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()
        self.annotaions = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.Transform = transform

    def __len__(self):
        return len(self.annotaions)

    def __getitem__(self, index):
        s = "cataract" if self.annotaions.iloc[index, 1] == 1 else "Normal"
        image_path = os.path.join(
            self.root_dir, s, self.annotaions.iloc[index, 0])
        image = plt.imread(image_path)
        y_label = torch.tensor(self.annotaions.iloc[index, 1])

        if self.Transform:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            image = self.Transform(image)
        return [image, y_label]


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
])


training = load_dataset(root_dir=training_image_path,
                        csv_file=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\C_N.csv', transform=train_transform)
val = load_dataset(root_dir=test_image_path,
                   csv_file=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\C_N_test_out.csv', transform=val_transform)
training_data = DataLoader(dataset=training, batch_size=32, shuffle=True)
val_data = DataLoader(dataset=val, batch_size=32, shuffle=True)
# print(next(iter(training_data))[1].shape)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


train(model=model,
      train_dataloader=training_data,
      test_dataloader=val_data,
      loss_fn=loss_fn,
      save_name="Cataract",
      optimizer=optimizer,
      epochs=5,
      print_freq=100)
