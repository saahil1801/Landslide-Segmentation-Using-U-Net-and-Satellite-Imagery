import streamlit as st
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Define the UNet architecture and its components
class DoubleConv(nn.Module):
    """Double Convolution Block"""
    def __init__(self, in_channels, out_channels, mid_channels=None, p=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling Block"""
    def __init__(self, in_channels, out_channels, p=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, p=p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling Block"""
    def __init__(self, in_channels, out_channels, bilinear=True, p=0.1):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, p=p)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, p=p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=6, n_classes=1, bilinear=True, p=0.1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 16, p=p)
        self.down1 = Down(16, 32, p=p)
        self.down2 = Down(32, 64, p=p)
        self.down3 = Down(64, 128, p=p)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor, p=p)

        self.up1 = Up(256, 128 // factor, bilinear, p=p)
        self.up2 = Up(128, 64 // factor, bilinear, p=p)
        self.up3 = Up(64, 32 // factor, bilinear, p=p)
        self.up4 = Up(32, 16, bilinear, p=p)
        self.outc = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)    # 16 channels
        x2 = self.down1(x1) # 32 channels
        x3 = self.down2(x2) # 64 channels
        x4 = self.down3(x3) # 128 channels
        x5 = self.down4(x4) # 256 channels

        x = self.up1(x5, x4)    # x5 upsampled and concatenated with x4
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        output = torch.sigmoid(logits)
        return output


# Load the trained model
device = 'mps'  # Set device to 'cpu' or 'mps', depending on your setup
model = UNet(n_channels=6, n_classes=1)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
model.to(device)

# Streamlit app
st.title("Landslide Segmentation Model")
st.write("Upload an image in `.h5` format to perform landslide segmentation.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image file", type=["h5"])

if uploaded_file is not None:
    st.write("Processing the image...")
    
    # Read the h5 image
    with h5py.File(uploaded_file, 'r') as hdf:
        data = np.array(hdf.get('img'))
        data[np.isnan(data)] = 0.000001  # Assign small value to NaNs

        # Normalize the data
        mid_rgb = data[:, :, 1:4].max() / 2.0
        mid_slope = data[:, :, 12].max() / 2.0
        mid_elevation = data[:, :, 13].max() / 2.0

        # NDVI calculation
        data_red = data[:, :, 3]
        data_nir = data[:, :, 7]
        data_ndvi = np.divide(data_nir - data_red, np.add(data_nir, data_red) + 1e-6)  # Avoid division by zero

        # Prepare the final input array
        image_tensor = np.zeros((1, 6, 128, 128))
        image_tensor[0, 0, :, :] = 1 - data[:, :, 3] / mid_rgb  # RED
        image_tensor[0, 1, :, :] = 1 - data[:, :, 2] / mid_rgb  # GREEN
        image_tensor[0, 2, :, :] = 1 - data[:, :, 1] / mid_rgb  # BLUE
        image_tensor[0, 3, :, :] = data_ndvi  # NDVI
        image_tensor[0, 4, :, :] = 1 - data[:, :, 12] / mid_slope  # SLOPE
        image_tensor[0, 5, :, :] = 1 - data[:, :, 13] / mid_elevation  # ELEVATION

        # Convert to torch tensor and move to device
        image_tensor = torch.from_numpy(image_tensor).float().to(device)

        # Disable gradient computation for inference
        with torch.no_grad():
            preds = model(image_tensor)
            threshold = 0.5
            preds_binary = (preds > threshold).float()

        # Detach the predictions and move to CPU for visualization
        predicted_mask = preds_binary[0].cpu().numpy()

        # Display the RGB image and predicted mask
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))

        # Display the RGB image
        axes[0].imshow(data[:, :, 0:3])
        axes[0].set_title('Validation Image')
        axes[0].axis('off')

        # Display the predicted mask
        axes[1].imshow(predicted_mask[0], cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')

        # Show the result
        st.pyplot(fig)

else:
    st.write("Please upload an image file to get predictions.")
