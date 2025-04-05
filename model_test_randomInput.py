import torch
import onnxruntime as ort
import numpy as np

# Set device for PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate a random input image (same for both models)
random_input = np.random.randn(1, 3, 32, 32).astype(np.float32)  

# Convert to PyTorch tensor
torch_input = torch.tensor(random_input).to(device)

# Load and Test PyTorch Model
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

# Load PyTorch Model
model = TrafficSignCNN().to(device)
model.load_state_dict(torch.load("traffic_sign_model.pth", map_location=device))
model.eval()

# Run inference with PyTorch model
torch_output = model(torch_input)

# Get predicted class
predicted_class_torch = torch.argmax(torch_output, dim=1).item()
print("✅ PyTorch Model Output:", torch_output)
print("✅ PyTorch Model Predicted Class:", predicted_class_torch)


# Load and Test ONNX Model
onnx_model_path = "traffic_sign_model.onnx"

# Create ONNX Runtime session
ort_session = ort.InferenceSession(onnx_model_path)

# Get input and output names
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Run inference
onnx_output = ort_session.run([output_name], {input_name: random_input})

# Convert ONNX output to PyTorch tensor for easy comparison
onnx_output_tensor = torch.tensor(onnx_output[0])

# Get predicted class from ONNX model
predicted_class_onnx = torch.argmax(onnx_output_tensor, dim=1).item()
print("ONNX Model Output:", onnx_output_tensor)
print("ONNX Model Predicted Class:", predicted_class_onnx)

# Compare Outputs
if predicted_class_torch == predicted_class_onnx:
    print("Both models predict the same class! ONNX conversion is successful.")
else:
    print("Different predictions! There may be an issue in ONNX conversion.")
