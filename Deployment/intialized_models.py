import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import cv2
import torch
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, efficientnet_b5, EfficientNet_B5_Weights, vgg16_bn, VGG16_BN_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch import nn
import torch
from torch import nn
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp

# # peprocessing
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def load_ben_color(image, sigmaX=30):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (224, 224))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(
        image, (0, 0), sigmaX), -4, 128)
    return Image.fromarray(image)


def preprocessing_on_image(dignosis):
    transform = None
    if dignosis == "D&B":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]
        )
    elif dignosis == "D":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
        )
    elif dignosis == "C":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        )
    elif dignosis == "G":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        )
    return transform


def initialized_model(dignosis, model_path):
    model = None
    if dignosis == "D&B":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(in_features=1280,
                                         out_features=2, bias=True)

    elif dignosis == "D":
        model = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=2048, out_features=5)

    elif dignosis == "C":
        model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        model.classifier = nn.Sequential(nn.Linear(25088, 64),
                                         nn.ReLU(),
                                         nn.Dropout(0.4),
                                         nn.Linear(64, 2)

                                         )
    elif dignosis == "G":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(in_features=1024, out_features=1)

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


def prediction(image, model_path, dignosis):

    model = initialized_model(dignosis, model_path)
    transform = preprocessing_on_image(dignosis)
    image = load_ben_color(image) if dignosis == "D" else image

    if transform:
        image = transform(image)

        image = image.unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        outputs = model(image.to(device))
        predicted = torch.argmax(outputs, dim=1) if dignosis != "G" else torch.round(
            torch.sigmoid(outputs))
    return predicted


# predicted = prediction(image_path=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\data\ODIR\On-site Test Set\Images\4886_left.jpg',
#                        model_path=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\models_pth\Cataract.pth', dignosis="C")
# print(predicted.item())
def model_1(image, model_path):
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,  # Since you'll load the model weights separately
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image = image / 255.0
    image_tensor = torch.tensor(
        image, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        pred_mask = model(image_tensor)
        pred_mask = torch.sigmoid(pred_mask).squeeze(0).squeeze(0)
        pred_mask = np.where(pred_mask.cpu().numpy() < 0.5,
                             0, 1).astype(np.uint8) * 255

    return pred_mask


# pred_mask = model_1(r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\data\ODIR\On-site Test Set\Images\4879_left.jpg',
#                     r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\models_pth\Segmentation_model.pth')
# print(type(pred_mask))
# plt.imshow(pred_mask)
# plt.show()
# Define the function for inference
