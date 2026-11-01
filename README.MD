# Feedback Emotion Analyzer

Deep Learning semester project (prototype).
The app reads user feedback and finds the emotion (7 classes) and the sentiment (positive, negative, neutral or ambiguous).

## Project Links
- Live app: https://feedback-emotion-analyzer-8ef6shsjrpcxzlkaz7ridr.streamlit.app/
- Project report: https://docs.google.com/document/d/1ALrzePsn29Koc6pLbs7yAQu3HrnZEwoMq11Lqi467Mo/edit
- Presentation: https://docs.google.com/presentation/d/1kpteRRXVeDgWR9eKlwvIdESI2DEjLB6V/edit
- YouTube video: ADD LATER
- LinkedIn post: ADD LATER
- Dataset: https://huggingface.co/datasets/google-research-datasets/go_emotions

## How it works
1. The text is cut into tokens (max length 64).
2. ELECTRA-small (a small transformer model) scores 7 emotions.
3. The emotions are grouped into a sentiment: positive, negative, neutral or ambiguous.
4. Streamlit shows the result, with a CSV batch analyzer for many feedbacks.

## Dataset
Secondary dataset: GoEmotions (Reddit comments labeled with emotions), loaded from Hugging Face.
The 28 original emotions were grouped into 7: anger, disgust, fear, joy, sadness, surprise and neutral.
Only comments whose labels fall in a single group were kept.
- Train: 39,555 comments
- Validation: 4,946 comments
- Test: 4,968 comments
See the dataset page for its license and credit.

## Model and results
- Model: google/electra-small-discriminator, fine-tuned (about 14 million parameters)
- Training: 4 epochs, batch size 64, learning rate 1e-4, max length 64, Google Colab T4 GPU
- Test accuracy: 0.70
- Test macro F1: 0.63
- Trained weights: sentiment_model/model.safetensors (about 54 MB, in this repo)

## Run it yourself
pip install -r requirements.txt
streamlit run app.py

## Files
- app.py: Streamlit web app
- train.py: training code
- sentiment_model/: trained model weights and tokenizer
- requirements.txt: Python packages

## Limitations
Trained on Reddit comments, so it can be less accurate on other writing styles. Sarcasm is hard for the model, and English only.
