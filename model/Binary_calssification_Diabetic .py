import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch import nn
import torch
from train_model import *


class DR_dataset(Dataset):
    def __init__(self, folder_path, annotation, transformation=None):
        super().__init__()
        self.source_path = folder_path
        self.annotation = annotation
        self.transformation = transformation

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        self.label = self.annotation.iloc[index]['diagnosis']
        id_image = self.annotation.iloc[index]['id_code']
        image_path = rf'{self.source_path}\{id_image}.png'
        self.image = Image.open(image_path)

        if self.transformation:
            self.image = self.transformation(self.image)
        return [self.image, self.label]


# load dataset then replace value of diagnosis to make it binary
df = pd.read_csv(r'D:\Graduated Project\DR\train.csv')
df['diagnosis'].replace({2: 1, 3: 1, 4: 1}, inplace=True)
df_train, df_test = train_test_split(
    df, test_size=0.33, random_state=42, shuffle=True)
# #
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
)
train_test_images = r'D:\Graduated Project\DR\train_images'
# test_images = r'D:\Graduated Project\DR\test_images'
train_dataset = DR_dataset(train_test_images, df_train, transform)
test_dataset = DR_dataset(train_test_images, df_test, transform)
# display result
trainig_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

model.classifier[-1] = nn.Linear(in_features=1280, out_features=2, bias=True)

model.load_state_dict(torch.load(
    r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\models_pth\Binary_DR.pth'))

# for child in list(model.children())[:-23]:
#     for param in child.parameters():
#         param.requires_grad=False

loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train(model=model,
      train_dataloader=trainig_loader,
      test_dataloader=trainig_loader,
      loss_fn=loss_fun,
      save_name="X",
      optimizer=optimizer,
      epochs=1,
      print_freq=100)
