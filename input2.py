from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# --- Hyperparameters ---
SENTIMENT_MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 32  # Token length (lowered for speed)
MODEL_SAVE_PATH = 'best_goemotions_model.pt'  # Path for saved model
SENSITIVITY_MODEL_NAME= 'facebook/bart-large-mnli'

# --- Load Tokenizer and Model ---
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME, num_labels=28)  # Number of emotions
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

# --- Fast Sensitivity Classifier ---
zero_shot_classifier = pipeline("zero-shot-classification", model=SENSITIVITY_MODEL_NAME) 
sensitive_labels = [
    "mental health", "depression", "stress", "suicide", "bullying", "eating disorder", 
    "self-harm", "grief", "loss of loved one", "domestic violence",

    "personal preference", "food preference", "hobby", "fashion", "entertainment"
]

for i in range(4):
    # --- Prediction Function ---
    def predict_emotion(text, model, tokenizer):
        # Tokenize the input text
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Apply sigmoid and threshold at 0.5
        probs = torch.sigmoid(logits)
        predictions = probs.cpu().numpy().flatten()  # Get the probabilities for each emotion

        emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
                    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
                    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
                    'relief', 'remorse', 'sadness', 'surprise', 'neutral']

        # Create a list of emotions with their corresponding probabilities
        emotion_probs = [(emotions[i], predictions[i]) for i in range(len(predictions))]

        # Sort the emotions by their probabilities (from highest to lowest)
        sorted_emotion_probs = sorted(emotion_probs, key=lambda x: x[1], reverse=True)

        return sorted_emotion_probs

    # --- Categorize Emotions ---
    positive_emotions = ['amusement', 'joy', 'approval', 'caring', 'curiosity', 'excitement', 'gratitude', 'love', 'optimism', 'pride', 'realization', 'relief']
    negative_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness', 'surprise']
    neutral_emotions = ['neutral']

    # --- Input and Prediction ---
    input_text = input("\n---------------------------------------\nEnter a sentence for emotion prediction: ")

    sorted_emotion_probs = predict_emotion(input_text, model, tokenizer)

    # --- Sentiment Score Calculation ---
    positive_total = 0
    negative_total = 0

    for emotion, prob in sorted_emotion_probs:
        if emotion in positive_emotions:
            positive_total += prob
        elif emotion in negative_emotions:
            negative_total += prob

    # Display the results
    print("\nPredicted emotions with rankings:")
    for rank, (emotion, prob) in enumerate(sorted_emotion_probs[:5], 1):  # Limiting to top 5
        print(f"Rank {rank}: {emotion} - Probability: {prob:.4f}")

    # Final score: positive - negative
    final_score = positive_total - negative_total
    print(f"\nFinal Sentiment Score (Positive - Negative): {final_score:.4f}\n")


    # --- Sentiment Warning Logic ---
    # if final_score > 0.7:
    #     sentiment_warning = "GREEN - Positive"
    # elif 0 < final_score <= 0.7:
    #     sentiment_warning = "YELLOW - Possibly Negative"
    # elif -0.7 <= final_score < 0:
    #     sentiment_warning = "ORANGE - Most likely Negative"
    # else:
    #     sentiment_warning = "RED - Negative"

    # --- Sensitivity Analysis ---
    sensitivity_output = zero_shot_classifier(input_text, candidate_labels=sensitive_labels)
    top_sensitive_label = sensitivity_output['labels'][0]


    for i in range(5):
        print(f"Rank {i+1}: {sensitivity_output['labels'][i]}")


# if top_sensitive_label in ["mental health", "depression", "stress", "suicide", "bullying", "eating disorder", "self-harm", "grief", "loss of loved one", "domestic violence"]:
#     sensitivity_warning = "most likely features sensitive topic: " + top_sensitive_label
#     trigger = 1
# else:
#     sensitivity_warning = "most likely features no sensitive topic"
#     trigger = 0

# if trigger == 1 and final_score < 0:
#     final = f"RED: Message most likely to be sensitive (topics regarding {top_sensitive_label}). A rewrite is strongly suggested."
# elif trigger == 1:
#     final = f"ORANGE: Message can be sensitive (topics regarding {top_sensitive_label}). A rewrite is suggested."
# elif trigger == 0 and final_score < 0:
#     final = "YELLOW: Message likely to have a negative tone. A rewrite is suggested."
# else:
#     final = "GREEN: Message likely to be safe."



# print(f"\n{final}\n")