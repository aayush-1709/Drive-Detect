import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import onnxruntime as ort

# Define Class Labels
class_labels = {
    0: "Speed limit 20",
    1: "Speed limit 30",
    2: "Speed limit 50",
    3: "Speed limit 60",
    4: "Speed limit 70",
    5: "Speed limit 80",
    # Add more if needed up to class 42
}

# Define the Model
class TrafficSignCNN(torch.nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Preprocessing Function
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  
    return image

# Load PyTorch Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrafficSignCNN().to(device)
model.load_state_dict(torch.load("traffic_sign_model.pth", map_location=device))
model.eval()

# Load ONNX Model
onnx_model_path = "traffic_sign_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Run Predictions on Folder
test_folder = r"C:\Users\1\Desktop\Sign Test"

for image_file in os.listdir(test_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(test_folder, image_file)
        input_tensor = preprocess_image(image_path)
        input_numpy = input_tensor.numpy().astype(np.float32)

        # PyTorch Prediction
        with torch.no_grad():
            output_pth = model(input_tensor.to(device))
            pred_pth = torch.argmax(output_pth, dim=1).item()

        # ONNX Prediction
        output_onnx = ort_session.run([output_name], {input_name: input_numpy})
        pred_onnx = np.argmax(output_onnx[0])

        # Get readable class names
        label_pth = class_labels.get(pred_pth, f"Unknown ({pred_pth})")
        label_onnx = class_labels.get(pred_onnx, f"Unknown ({pred_onnx})")

        # Print results
        print(f"📷 {image_file}")
        print(f"PyTorch Prediction: {pred_pth} → {label_pth}")
        print(f"ONNX Prediction:    {pred_onnx} → {label_onnx}")
        if pred_pth == pred_onnx:
            print("Match\n")
        else:
            print("Mismatch\n")
