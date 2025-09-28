import cv2
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Configuration
class Config:
    EMO_NUM_CLASSES = 7
    EMO_MODEL_PATH = 'mobilevit_emotion_best.pth'
    EMO_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    RACE_NUM_CLASSES = 5
    RACE_MODEL_PATH = 'race.pth'
    RACE_CLASSES = ['Asian', 'Black', 'Indian', 'Others', 'White']
    GENDER_NUM_CLASSES = 2
    GENDER_MODEL_PATH = 'gender.pth'
    GENDER_CLASSES = ['female', 'male']
    AGE_NUM_CLASSES = 10
    AGE_MODEL_PATH = 'age.pth'
    AGE_CLASSES = [f"{i*10}-{i*10+9}" for i in range(AGE_NUM_CLASSES)]
    IMG_SIZE = 112
    RGB_SIZE = 48
    GRAY_MEAN = 0.5
    GRAY_STD = 0.5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# MobileViTEmotion Model
class MobileViTEmotion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            'mobilevit_s',
            pretrained=False,
            features_only=True,
            in_chans=1
        )
        self.conv1 = nn.Conv2d(640, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.shortcut = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        x = features[-1]
        x1 = F.silu(self.bn1(self.conv1(x)))
        identity = self.shortcut(x)
        x2 = F.silu(self.bn2(self.conv2(x1)))
        x3 = self.bn3(self.conv3(x2))
        x3 += identity
        x = F.silu(x3)
        att = self.attention(x)
        x = x * att
        return self.classifier(x)

# Face Analyzer
class FaceAnalyzer:
    def __init__(self):
        # Load models
        self.emo_model = self._load_emotion_model(Config.EMO_NUM_CLASSES, Config.EMO_MODEL_PATH)
        self.race_model = self._load_generic_model(Config.RACE_NUM_CLASSES, Config.RACE_MODEL_PATH, in_chans=3, img_size=Config.RGB_SIZE)
        self.gender_model = self._load_generic_model(Config.GENDER_NUM_CLASSES, Config.GENDER_MODEL_PATH, in_chans=3, img_size=Config.RGB_SIZE)
        self.age_model = self._load_generic_model(Config.AGE_NUM_CLASSES, Config.AGE_MODEL_PATH, in_chans=3, img_size=Config.RGB_SIZE)
        
        # Preprocessing
        self.emo_preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([Config.GRAY_MEAN], [Config.GRAY_STD])
        ])
        self.rgb_preprocess = transforms.Compose([
            transforms.Resize((Config.RGB_SIZE, Config.RGB_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(Config.CASCADE_PATH)
        if self.face_cascade.empty():
            raise IOError(f"Failed to load cascade classifier: {Config.CASCADE_PATH}")
        self.colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

    def _load_emotion_model(self, num_classes, model_path):
        model = MobileViTEmotion(num_classes)
        state = torch.load(model_path, map_location=Config.DEVICE)
        if all(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(Config.DEVICE)
        model.eval()
        return model

    def _load_generic_model(self, num_classes, model_path, in_chans, img_size):
        model = timm.create_model(
            'mobilevit_s',
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_chans,
            img_size=img_size
        )
        state = torch.load(model_path, map_location=Config.DEVICE)
        if all(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(Config.DEVICE)
        model.eval()
        return model

    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48,48))
        results = []
        for idx, (x, y, w, h) in enumerate(faces):
            face_roi = frame[y:y+h, x:x+w]
            with torch.no_grad():
                emo_input = self.emo_preprocess(Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)).convert('L')).unsqueeze(0).to(Config.DEVICE)
                emo_logits = self.emo_model(emo_input)
                emo_idx = emo_logits.argmax(1).item()
                emo_conf = F.softmax(emo_logits, dim=1)[0, emo_idx].item()
                
                rgb_input = self.rgb_preprocess(Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(Config.DEVICE)
                race_idx = self.race_model(rgb_input).argmax(1).item()
                gender_idx = self.gender_model(rgb_input).argmax(1).item()
                age_idx = self.age_model(rgb_input).argmax(1).item()
            results.append({
                'bbox': (x, y, w, h),
                'emo': f"{Config.EMO_CLASSES[emo_idx]} ({emo_conf*100:.1f}%)",
                'race': Config.RACE_CLASSES[race_idx],
                'gender': Config.GENDER_CLASSES[gender_idx],
                'age': Config.AGE_CLASSES[age_idx],
                'color': self.colors[idx % len(self.colors)]
            })
        return results

    def visualize_results(self, frame, results):
        for result in results:
            x, y, w, h = result['bbox']
            color = result['color']
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            info = f"{result['emo']} | {result['gender']} | {result['age']}"
            cv2.putText(frame, info, (x, y-10 if y-10>10 else y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Race: {result['race']}", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

def main():
    analyzer = FaceAnalyzer()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow('Face Analysis', cv2.WINDOW_NORMAL)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            frame = cv2.flip(frame, 1)
            results = analyzer.analyze_frame(frame)
            visualized_frame = analyzer.visualize_results(frame, results)
            cv2.imshow('Face Analysis', visualized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
