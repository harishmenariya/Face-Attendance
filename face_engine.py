import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image

class MobileNetV2Embedder(nn.Module):
    def __init__(self):
        super(MobileNetV2Embedder, self).__init__()
        # Load base model without weights
        base_model = models.mobilenet_v2(weights=None)
        
        # Feature extractor
        self.features = base_model.features
        
        # Pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom head
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(base_model.last_channel, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        # Extract features
        x = self.features(x)
        # Apply adaptive pooling
        x = self.pool(x)
        # Flatten
        x = torch.flatten(x, 1)
        # Pass through head
        x = self.head(x)
        # Apply L2 normalization
        x = F.normalize(x, p=2, dim=1)
        return x


class FaceEngine:
    def __init__(self, 
                 detector_path='models/face_detection_yunet_2023mar.onnx',
                 embedder_path='models/mobilenetv2_feature_extractor.pth'):
        # Initialize device strictly on CPU
        self.device = torch.device("cpu")
        
        # Load YuNet Detector
        if not os.path.exists(detector_path):
            print(f"[Warning] Detector model not found at {detector_path}. Detection will fail if called.")
            self.detector = None
        else:
            self.detector = cv2.FaceDetectorYN.create(
                detector_path,
                "",
                (640, 480),
                score_threshold=0.8,
                nms_threshold=0.3
            )
            
        # Load MobileNetV2 Embedder
        self.embedder = MobileNetV2Embedder()
        if os.path.exists(embedder_path):
            self.embedder.load_state_dict(torch.load(embedder_path, map_location=self.device))
        else:
            print(f"[Warning] Embedder weights not found at {embedder_path}. Using random weights.")
            
        self.embedder.to(self.device)
        self.embedder.eval()
        
        # Image Preprocessing Transform
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_faces(self, frame_bgr):
        if self.detector is None:
            raise RuntimeError("Detector model was not loaded properly.")
            
        # Dynamically set input size based on current frame shape
        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        
        # Detect faces
        _, faces = self.detector.detect(frame_bgr)
        return faces

    def extract_embedding(self, face_crop_bgr) -> np.ndarray:
        # Convert BGR crop to RGB
        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        
        # Create PIL Image
        pil_img = Image.fromarray(face_crop_rgb)
        
        # Apply transforms
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Run forward pass through the embedder
        with torch.no_grad():
            emb = self.embedder(tensor_img)
            
        # Return a 128-dimensional float32 NumPy array
        return emb.squeeze().cpu().numpy().astype(np.float32)


if __name__ == '__main__':
    print("=== STEP 4: face_engine.py SELF-TEST ===")
    
    try:
        # 1. Initialize FaceEngine
        engine = FaceEngine()
        
        # 2. Feed a synthetic mock frame through detection
        mock_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        print("Feeding mock frame of shape:", mock_frame.shape)
        if engine.detector is not None:
            faces = engine.detect_faces(mock_frame)
            print("Detected faces:", faces if faces is not None else "None")
        else:
            print("Skipped face detection test due to missing model.")
            
        # 3. Run a dummy face crop through extract_embedding
        mock_face_crop = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        embedding = engine.extract_embedding(mock_face_crop)
        
        # 4. Confirm the output shape is (128,)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding dtype: {embedding.dtype}")
        
        if embedding.shape == (128,):
            print("[OK] Embedding extraction successful!")
        else:
            print("[ERROR] Embedding shape mismatch.")
            
    except Exception as e:
        print(f"[ERROR] Self-test failed: {e}")
    print("=== SELF-TEST COMPLETE ===")
