import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# Load and Preprocess Image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),       
        transforms.ToTensor(),             
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0) 
    
    return image_tensor

# Example image path (upload your image and replace this path)
image_path = r"C:\Users\1\Desktop\Dyne\Project\gtsrb-german-traffic-sign\versions\1\Train\28\00028_00000_00023.png"
input_tensor = preprocess_image(image_path)

# Convert to numpy for ONNX
input_numpy = input_tensor.numpy().astype(np.float32)

# Load and Predict with PyTorch Model
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

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrafficSignCNN().to(device)
model.load_state_dict(torch.load("traffic_sign_model.pth", map_location=device))
model.eval()

# Predict using PyTorch model
with torch.no_grad():
    output_pth = model(input_tensor.to(device))
    pred_pth = torch.argmax(output_pth, dim=1).item()

print("✅ PyTorch Model Prediction:", pred_pth)

# Load and Predict with ONNX Model
onnx_session = ort.InferenceSession("traffic_sign_model.onnx")
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

output_onnx = onnx_session.run([output_name], {input_name: input_numpy})
pred_onnx = np.argmax(output_onnx[0])

print("✅ ONNX Model Prediction:", pred_onnx)

# Compare Results
if pred_pth == pred_onnx:
    print("Both models predict the same class!")
else:
    print("Different predictions!")
