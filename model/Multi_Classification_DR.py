import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b5, EfficientNet_B5_Weights
from torch import nn
import torch
from train_model import *
import cv2


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
    return img


def load_ben_color(path, sigmaX=30):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (224, 224))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(
        image, (0, 0), sigmaX), -4, 128)
    return image


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
        # self.image = Image.open(image_path)

        self.image = load_ben_color(image_path)

        if self.transformation:
            self.image = Image.fromarray(self.image.astype(np.uint8))
            self.image = self.transformation(self.image)
        return [self.image, self.label]


df = pd.read_csv(r'D:\Graduated Project\DR\train.csv')
df_train, df_test = train_test_split(
    df, test_size=0.33, random_state=42, shuffle=True)

transform_train = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),       # Random horizontal flip
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
]
)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
]
)
train_test_images = r'D:\Graduated Project\DR\train_images'
# test_images = r'D:\Graduated Project\DR\test_images'
train_dataset = DR_dataset(train_test_images, df_train, transform_train)
test_dataset = DR_dataset(train_test_images, df_test, transform)
# display result
trainig_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
model = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(in_features=2048, out_features=5)
model.load_state_dict(torch.load(
    r'D:\Graduated Project\DR\modelv2.pth', map_location=device))


loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train(model=model,
      train_dataloader=trainig_loader,
      test_dataloader=test_loader,
      loss_fn=loss_fun,
      optimizer=optimizer,
      save_name="MultiDR",
      epochs=10,
      print_freq=100)
