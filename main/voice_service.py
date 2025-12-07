import joblib
import numpy as np
import librosa
import os
import whisper
from scipy.stats import skew, kurtosis

class VoiceAnalysisService:
    def __init__(self):
        # 1. Load Chat Model (Whisper)
        print("🎤 Loading Whisper (Speech-to-Text)...")
        try:
            self.whisper_model = whisper.load_model("base")
        except Exception as e:
            print(f"❌ Error loading Whisper: {e}")
            self.whisper_model = None

        # 2. Load Medical Models (Sickness Detection)
        print("🩺 Loading Medical Diagnosis Models...")
        self.model_screener = None
        self.model_specialist = None
        self.threshold = 0.22 
        
        # Adjust paths if needed
        self.base_path = r"C:\Users\khali\OneDrive\Bureau\medical_rag\voice-model"
        self.screener_path = os.path.join(self.base_path, "model_screener.pkl")
        self.specialist_path = os.path.join(self.base_path, "model_specialist.pkl")
        self.threshold_path = os.path.join(self.base_path, "threshold_screener.txt")
        
        self.load_medical_models()

    def load_medical_models(self):
        try:
            if os.path.exists(self.screener_path):
                self.model_screener = joblib.load(self.screener_path)
            if os.path.exists(self.specialist_path):
                self.model_specialist = joblib.load(self.specialist_path)
            if os.path.exists(self.threshold_path):
                with open(self.threshold_path, 'r') as f:
                    self.threshold = float(f.read().strip())
            print(f"✅ Medical Models Loaded (Sensitivity: {self.threshold:.1%})")
        except Exception as e:
            print(f"❌ Error loading medical models: {e}")

    # --- FEATURE 1: Chat Transcription ---
    def transcribe(self, audio_path):
        """Converts audio file to text using Whisper"""
        if not self.whisper_model:
            return "Error: Whisper model not loaded."
        
        result = self.whisper_model.transcribe(audio_path)
        return result["text"].strip()

    # --- FEATURE 2: Medical Diagnosis ---
    def extract_features(self, file_path):
        """Extracts 16 features from audio"""
        try:
            y, sr = librosa.load(file_path, sr=22050, duration=4.0)
            y, _ = librosa.effects.trim(y)
            if len(y) < 1024: return np.zeros(16)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            delta = librosa.feature.delta(mfcc)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

            def stats(m):
                if m.size == 0: return [0, 0, 0, 0]
                return [np.mean(m), np.var(m), skew(m, axis=None), kurtosis(m, axis=None)]

            features = []
            for x in [mfcc, delta, cent, contrast]: features.extend(stats(x))
            return np.array(features)
        except: return np.zeros(16)

    def diagnose(self, cough_path, breath_path, speech_path):
        """Runs the 2-Stage Sickness Detection"""
        if not self.model_screener:
            return {"status": "ERROR", "details": "Models not loaded"}

        # Extract features from all 3 files
        feats = []
        feats.extend(self.extract_features(cough_path))
        feats.extend(self.extract_features(breath_path))
        feats.extend(self.extract_features(speech_path))
        
        X = np.array(feats).reshape(1, -1)

        # Stage 1: Screen
        sick_prob = self.model_screener.predict_proba(X)[:, 1][0]
        
        result = {
            "sickness_prob": float(sick_prob),
            "status": "HEALTHY",
            "details": "Biomarkers normal."
        }

        # Stage 2: Specialist
        if sick_prob >= self.threshold:
            # Check if specialist model exists, otherwise just report sickness
            if self.model_specialist:
                covid_prob = self.model_specialist.predict_proba(X)[:, 1][0]
                result["covid_prob"] = float(covid_prob)
                
                if covid_prob > 0.50:
                    result["status"] = "COVID-19 DETECTED"
                    result["details"] = "Matches COVID profile."
                else:
                    result["status"] = "RESPIRATORY ISSUE"
                    result["details"] = "Non-COVID sickness signs detected."
            else:
                 result["status"] = "SICKNESS DETECTED"
                 result["details"] = "Irregularities found."

        return result

voice_engine = VoiceAnalysisService()