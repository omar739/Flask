#-----------------------------------------------------------------
from flask import Flask, render_template, request
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import random as rd
import torchvision.models as models
from torchvision.transforms.functional import to_pil_image
import cv2
#-----------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.bottleneck = double_conv(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2, stride=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2, stride=2))
        e4 = self.enc4(F.max_pool2d(e3, kernel_size=2, stride=2))
        b = self.bottleneck(F.max_pool2d(e4, kernel_size=2, stride=2))
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        return self.out_conv(d1)
#-----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("best_model_final.pth"))
#-----------------------------------------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#-----------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def upload_image():
    imagee = None
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"
        file = request.files["image"]
        target = request.form["Target"]
        if file.filename == "":
            return "No selected file"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        processed_image_path = process_target(filepath, target)
        if processed_image_path:
            imagee = processed_image_path
    return render_template("filee.html", imagee=imagee)
#-----------------------------------------------------------------
def Normalize(image):
    return (((image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)) * 255).astype(np.uint8)
def process_target(image_file, target):
    with rasterio.open(image_file) as src:
        image = src.read()
    if target == "Band1":
        band1 = Normalize(image[0, :, :])
        plt.figure(figsize=(6, 6))
        sns.heatmap(band1)
        plt.axis("off")
        heatmap_path = "static/uploads/heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return "uploads/heatmap.png"
    elif target== "RGB" :
        green = np.array(image[2,:,:])
        blue  = np.array(image[1,:,:])
        red   = np.array(image[3,:,:])
        RGB = Normalize(np.stack((red,green,blue),axis=2))
        plt.figure(figsize=(6,6))
        plt.imshow(RGB)
        plt.axis('off')
        rgb_path = "static/uploads/RGB.png"
        plt.savefig(rgb_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return "uploads/RGB.png"
    elif target== "Band5" :
        band5 = Normalize(image[4, :, :])
        plt.figure(figsize=(6, 6))
        sns.heatmap(band5)
        plt.axis("off")
        heatmap_path = "static/uploads/heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return "uploads/heatmap.png"
    elif target== "Band6" :
        band6 = Normalize(image[5, :, :])
        plt.figure(figsize=(6, 6))
        sns.heatmap(band6)
        plt.axis("off")
        heatmap_path = "static/uploads/heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return "uploads/heatmap.png"
    elif target== "Band7" :
        band7 = Normalize(image[6, :, :])
        plt.figure(figsize=(6, 6))
        sns.heatmap(band7)
        plt.axis("off")
        heatmap_path = "static/uploads/heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return "uploads/heatmap.png"
    elif target== "Band8" :
        band8 = Normalize(image[7, :, :])
        plt.figure(figsize=(6, 6))
        sns.heatmap(band8)
        plt.axis("off")
        heatmap_path = "static/uploads/heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return "uploads/heatmap.png"
    elif target== "Band9" :
        band9 = Normalize(image[8, :, :])
        plt.figure(figsize=(6, 6))
        plt.imshow(band9,cmap="terrain")
        plt.axis("off")
        heatmap_path = "static/uploads/heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return "uploads/heatmap.png"
    elif target== "Band10" :
        band10 = Normalize(image[9, :, :])
        plt.figure(figsize=(6, 6))
        plt.imshow(band10,cmap="terrain")
        plt.axis("off")
        heatmap_path = "static/uploads/heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return "uploads/heatmap.png"
    elif target== "Band11" :
        band11 = Normalize(image[10, :, :])
        plt.figure(figsize=(6, 6))
        sns.heatmap(band11)
        plt.axis("off")
        heatmap_path = "static/uploads/heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return "uploads/heatmap.png"
    elif target== "Band12" :
        band12 = Normalize(image[11, :, :])
        plt.figure(figsize=(6, 6))
        sns.heatmap(band12)
        plt.axis("off")
        heatmap_path = "static/uploads/heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return "uploads/heatmap.png"
    elif target== "Detect Water" :
        return predict_new_image(image)
#-----------------------------------------------------------------
def normalize_band(band):
    return (band - band.min()) / (band.max() - band.min())
#-----------------------------------------------------------------
def preprocessing_image(np_comb):
    np_comb = np_comb.astype(np.float32)
    if np_comb.max() > 1.0:
        np_comb /= 255.0
    pil_image = to_pil_image(np_comb)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(pil_image)
#-----------------------------------------------------------------
def predict_new_image(image):
    print("Image shape before selecting bands:", image.shape)
    band_comb1 = normalize_band(image[4, :, :])
    band_comb2 = normalize_band(image[10, :, :])
    band_comb3 = normalize_band(image[3, :, :])
    Comb_All = np.stack((band_comb1, band_comb2, band_comb3), axis=2)
    Comb_All_preprocessed = preprocessing_image(Comb_All)
    Comb_All_preprocessed = Comb_All_preprocessed.to(device).unsqueeze(0)
    with torch.no_grad():
        output = model(Comb_All_preprocessed)
    output = (output).cpu().numpy().squeeze()
    output_uint8 = (output * 255).astype(np.uint8)
    _, output_mask = cv2.threshold(output_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure(figsize=(6, 6))
    plt.imshow(output_mask, cmap="gray")
    plt.axis("off")
    output_path = "static/uploads/predict.png"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return "uploads/predict.png"
#-----------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
