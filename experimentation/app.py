import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os

# --- 1. Model Architecture (matches the trained .pth) ---
class TrafficSignCNN(nn.Module):
    def __init__(self):
        super(TrafficSignCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512) 
        self.fc2 = nn.Linear(512, 43)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Class Mapping ---
CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# --- 3. Streamlit Logic ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(page_title="Drive-Detect AI", page_icon="🚦")

@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficSignCNN().to(device)
    model_path = os.path.join(current_dir, "traffic_sign_model.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    return None, device

st.title("🚦 Drive-Detect: Traffic Sign AI")
st.markdown("Upload a photo of a traffic sign for real-time Deep Learning classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('🚀 Run Inference'):
        model, device = load_model()
        if model is None:
            st.error("Model weights (.pth) not found in the experimentation folder!")
        else:
            with st.spinner('Analyzing...'):
                # Preprocessing
                img = np.array(image.convert('RGB'))
                img = cv2.resize(img, (32, 32))
                img = img / 255.0
                img = (img - 0.5) / 0.5
                img_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(prob, 1)
                
                label = CLASSES.get(predicted.item(), "Unknown")
                st.divider()
                st.subheader(f"Result: **{label}**")
                st.write(f"Confidence: {confidence.item()*100:.2f}%")
                st.progress(confidence.item())