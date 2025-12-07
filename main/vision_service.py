import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
import os

# --- CONFIG ---
# Update this to where your model file actually is
MODEL_PATH = r"C:\Users\khali\OneDrive\Bureau\medical_rag\chest\chexpert_model_2gpu.pth" 
IMG_SIZE = 320
LABELS = np.array([
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 
    'Support Devices'
])

# --- MODEL DEFINITION ---
class CheXpertClassifier(nn.Module):
    def __init__(self, num_classes=14):
        super(CheXpertClassifier, self).__init__()
        self.backbone = models.densenet121(weights=None)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone.features(x)
        out = F.relu(features, inplace=False) 
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.backbone.classifier(out)
        return out

# --- GRAD-CAM ENGINE ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self):
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        return heatmap

# --- SERVICE CLASS ---
class VisionService:
    def __init__(self):
        print("👁️ Loading Vision Model (CheXpert)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️ Warning: Vision model not found at {MODEL_PATH}")
            return

        try:
            self.model = CheXpertClassifier(num_classes=14)
            # Handle the 'module.' prefix if trained on multiple GPUs
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Vision Model Ready!")
        except Exception as e:
            print(f"❌ Error loading vision model: {e}")

    def process_image(self, image_bytes):
        if not self.model:
            return {"error": "Model not loaded"}

        try:
            # 1. Preprocess
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(pil_img).unsqueeze(0).to(self.device)
            img_tensor.requires_grad = True

            # 2. Grad-CAM Setup
            target_layer = self.model.backbone.features.norm5
            grad_cam = GradCAM(self.model, target_layer)

            # 3. Predict
            self.model.zero_grad()
            output = self.model(img_tensor)
            probs = torch.sigmoid(output).cpu().detach().numpy()[0]

            # 4. Get Top Prediction
            sorted_indices = np.argsort(probs)
            top_idx = sorted_indices[-1]
            top_label = LABELS[top_idx]
            top_prob = probs[top_idx]

            # 5. Generate Heatmap
            output[:, top_idx].backward()
            heatmap = grad_cam.generate_heatmap()

            # 6. Visualization (Matplotlib -> Base64)
            # Resize heatmap to match original image
            original_cv = np.array(pil_img)
            heatmap = cv2.resize(heatmap, (original_cv.shape[1], original_cv.shape[0]))
            heatmap[heatmap < 0.2] = 0 # Threshold
            
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_INFERNO)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(original_cv, 0.7, heatmap_colored, 0.3, 0)

            # Plotting
            fig = plt.figure(figsize=(12, 6))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

            # Panel A: Heatmap
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(overlay)
            ax1.set_title(f"Focus: {top_label}", fontsize=14, fontweight='bold')
            ax1.axis('off')

            # Panel B: Bar Chart
            ax2 = fig.add_subplot(gs[0, 1])
            # Filter top 5 for cleaner UI
            top_5_indices = sorted_indices[-5:]
            top_5_labels = LABELS[top_5_indices]
            top_5_probs = probs[top_5_indices]
            
            colors = ['#e74c3c' if x > 0.5 else '#2ecc71' for x in top_5_probs]
            bars = ax2.barh(top_5_labels, top_5_probs, color=colors)
            ax2.set_xlim(0, 1.05)
            ax2.set_title("Top 5 Findings", fontsize=14, fontweight='bold')
            
            # Add labels
            for bar, p in zip(bars, top_5_probs):
                ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                         f"{p:.1%}", va='center', fontsize=10)

            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            return {
                "status": "success",
                "top_finding": top_label,
                "confidence": float(top_prob),
                "image_base64": img_str
            }

        except Exception as e:
            print(f"Vision Error: {e}")
            return {"status": "error", "message": str(e)}

vision_engine = VisionService()