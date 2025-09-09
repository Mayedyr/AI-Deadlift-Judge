import torch
import torch.nn as nn
import os
from pytorch_tcn import TCN
from ultralytics import YOLO
import streamlit as st

# Initialize device and posemodel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
posemodel = YOLO("yolo11x-pose.pt")

# Model Definition
class DeadliftTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, kernel_size=2, dropout=0.2):
        super(DeadliftTCN, self).__init__()
        self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x): # x shape: (batch_size, input_size, sequence_length)
        y = self.tcn(x)  # Shape: (batch, num_channels[-1], sequence_length)
        y = torch.mean(y, dim=2)  # Global average pooling over time dimension
        logits = self.fc(y)       # Raw logits
        return logits
    
# Initialize Model and Hyperparameters
input_size = 34 # 17 joints * 2 coordinates per frame, flattened
num_classes = 3 # 3 infractions
num_channels = [50,75,100,125,125] # Filters per layer
kernel_size = 7
dropout = 0.5
model = DeadliftTCN(input_size, num_channels, num_classes, kernel_size, dropout).to(device)
model.load_state_dict(torch.load("model_weights.pt", map_location=device, weights_only=True))
model.eval()  # Set the model to evaluation mode

def pad_sequence(tensor, max_length=200):
    # Pad or truncate the input tensor along the first dimension to a fixed length
    num_frames = tensor.shape[0]
    if num_frames < max_length:
        padding = torch.zeros(max_length - num_frames, tensor.shape[1], tensor.shape[2], device=tensor.device)
        return torch.cat([tensor, padding], dim=0)
    return tensor[:max_length]

def process_video_file(video_path):
    #Processes the video file using the posemodel to extract normalized keypoints.
    tensors_list = []
    # Use the posemodel in streaming mode to process the video file
    results = posemodel(video_path, iou=0.4, verbose=False, max_det=1, stream=True)
    for result in results:
        try:
            keypoints = result.keypoints.xyn.reshape(17, 2)
            tensors_list.append(keypoints)
        except Exception as e:
            pass
    if not tensors_list:
        return None
    video_tensor = torch.stack(tensors_list)  # shape: (num_frames, 17, 2)
    video_tensor = pad_sequence(video_tensor, max_length=200)
    return video_tensor

def describe_prediction(prediction):
    #Converts a list of three binary predictions into a readable description.
    labels = ["🔴 Red Card: Soft Lockout",
              "🔵 Blue Card: Downwards Movement / Support on Thighs",
              "🟡 Yellow Card: Other Infraction / Incomplete Lift"]
    desc_parts = []
    for label, pred in zip(labels, prediction):
        if pred == 1:
            st.write(f"— {label}: detected")
    
    overall = "Good Lift! ⚪⚪⚪" if sum(prediction) == 0 else "No Lift! ❌❌❌" # Determine overall Judgement
    st.write(overall)

# Streamlit App Layout
st.title("Deadlift Judge Prediction App")
st.write("Upload a video file (MP4) and get a prediction from the model.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = "temp_video.mp4"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(uploaded_file)
    
    st.write("Processing video and making predictions...")
    video_tensor = process_video_file(temp_file_path)
    
    if video_tensor is None:
        st.error("Failed to process video. Please try another file.")
    else:
        # (200, 17, 2) -> (200, 34) -> (34, 200)
        video_tensor = video_tensor.view(200, -1).transpose(0, 1)  # shape: (34, 200)
        video_tensor = video_tensor.unsqueeze(0).to(device)  # add batch dimension -> (1, 34, 200)
        
        with torch.no_grad():
            logits = model(video_tensor)
            probs = torch.sigmoid(logits)
            # Set threshold for binary classification on each label:
            prediction = (probs > 0.9).float().cpu().tolist()
        
        describe_prediction(prediction[0])
    
    # Clean up temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)