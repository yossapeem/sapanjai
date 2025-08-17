from dotenv import load_dotenv
import os
from cog import BasePredictor, Input
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import snapshot_download
import json

load_dotenv()  # Load environment variables from .env

class Predictor(BasePredictor):
    def setup(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.SENTIMENT_MODEL_NAME = 'distilbert-base-uncased'
        self.MAX_LEN = 32
        self.MODEL_SAVE_PATH = 'best_goemotions_model.pt'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.SENTIMENT_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.SENTIMENT_MODEL_NAME,
            num_labels=28
        )
        self.model.load_state_dict(torch.load(self.MODEL_SAVE_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Load fast sensitivity classifier embeddings
        snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            resume_download=True,
        )
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sensitive_labels = [
            "mental health", "depression", "stress", "suicide", "self-harm", "anxiety",
            "bullying", "harassment", "abuse", "physical abuse", "emotional abuse", "sexual abuse",
            "addiction", "eating disorder", "trauma", "grief", "loss", "domestic violence", "violence",
            "discrimination", "hate speech", "toxic language", "relationship breakup",
            
            "personal preference", "food preference", "hobby", "fashion", "entertainment",
            "love", "gratitude", "friendship", "relationship", "career", "education",
            "health", "fitness", "travel", "technology", "pets", "music", "movies"
        ]
        self.label_embeddings = self.embed_model.encode(self.sensitive_labels, convert_to_tensor=True)

        self.positive_emotions = [
            'amusement', 'joy', 'approval', 'caring', 'curiosity', 'excitement',
            'gratitude', 'love', 'optimism', 'pride', 'realization', 'relief'
        ]
        self.negative_emotions = [
            'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
            'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness', 'surprise'
        ]

        self.critical_sensitive = {
            "mental health", "depression", "stress", "suicide", "self-harm", "anxiety",
            "bullying", "harassment", "abuse", "physical abuse", "emotional abuse", "sexual abuse",
            "addiction", "eating disorder", "trauma", "grief", "loss", "domestic violence", "violence",
            "discrimination", "hate speech", "toxic language", "relationship breakup"
        }

    def fast_sensitivity_analysis(self, text):
        text_emb = self.embed_model.encode(text, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(text_emb, self.label_embeddings).squeeze()
        top_indices = similarities.argsort(descending=True)
        return {
            'labels': [self.sensitive_labels[i] for i in top_indices],
            'scores': [similarities[i].item() for i in top_indices]
        }

    def predict_emotion(self, text: str):
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        emotions = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        emotion_probs = [(emotions[i], probs[i]) for i in range(len(emotions))]
        return sorted(emotion_probs, key=lambda x: x[1], reverse=True)

    def predict(self, text: str = Input(description="Input text to analyze")) -> dict:

        response = self.generate_supportive_response(text)
        input_text = response.get("original_text", text)
    
        # Emotion prediction
        sorted_emotion_probs = self.predict_emotion(input_text)

        positive_total = sum(prob for emo, prob in sorted_emotion_probs if emo in self.positive_emotions)
        negative_total = sum(prob for emo, prob in sorted_emotion_probs if emo in self.negative_emotions)
        final_score = positive_total - negative_total

        # Sensitivity analysis
        sensitivity_output = self.fast_sensitivity_analysis(input_text)
        top_sensitive_label = sensitivity_output['labels'][0]
        top_sensitive_score = sensitivity_output['scores'][0]
        is_critical = top_sensitive_label in self.critical_sensitive

        # Sentiment classification
        if is_critical and final_score < 0:
            sentiment_level = "red"
            final_advice = f"RED: Message most likely to be sensitive (topics regarding {top_sensitive_label}). A rewrite is strongly suggested."
        elif is_critical:
            sentiment_level = "orange"
            final_advice = f"ORANGE: Message can be sensitive (topics regarding {top_sensitive_label}). A rewrite is suggested."
        elif final_score < 0:
            sentiment_level = "yellow"
            final_advice = "YELLOW: Message likely to have a negative tone. A rewrite is suggested."
        else:
            sentiment_level = "green"
            final_advice = "GREEN: Message likely to be safe."

        return {
            "original_text": text,
            "input_text": input_text,
            "sentiment_level": sentiment_level,
            "final_advice": final_advice,
        }

    def generate_supportive_response(self, user_message):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are a translator and sentiment analyzer. "
                    "Your job has two steps: "
                    "1. If input is Thai (or contains Thai), translate it to English. "
                    "   Replace Thai bad words with their English equivalents. "
                    "2. Return ONLY a JSON object in this format:\n"
                    "{\n"
                    "  \"input_text\": \"<user input as given>\",\n"
                    "  \"original_text\": \"<English version>\",\n"
                    "  \"sentiment_level\": \"green|yellow|orange|red\",\n"
                    "  \"final_advice\": \"<advice sentence>\"\n"
                    "}\n"
                    "Do not include explanations or any extra text. "
                )},
                {"role": "user", "content": user_message},
            ],
        )
        raw_output = completion.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            parsed = {"error": "Invalid JSON", "raw": raw_output}
        return parsed
