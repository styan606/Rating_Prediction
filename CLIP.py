from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from io import BytesIO
import requests
import pandas as pd
import torch
import matplotlib.pyplot as plt
from translatepy import Translator
import os

# Initialize the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Translator for language detection and translation
translator = Translator()


def fetch_image(url):
    """Fetch image without resizing and ensure it is in RGB mode."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")  # Explicitly ensure RGB mode
        image = image.resize((image.width, image.height))  # Force proper shape processing
        return image
    except Exception as e:
        print(f"Failed to fetch image from {url}: {e}")
        return None



def fetch_image_resize(url):
    """Fetch and resize image to 224x224."""
    try:
        image = fetch_image(url)
        if image is not None:
            image = image.resize((224, 224))  # Resize to fixed dimensions
        return image
    except Exception as e:
        print(f"Failed to resize image from {url}: {e}")
        return None

def translate_to_english(text):
    """Detect if the text is non-English and translate it to English."""
    try:
        translation = translator.translate(text, "English")
        if translation.source_language != "English":
            #print(f"Translated '{text}' to English: {translation.result}")
            return translation.result
        return text  # Already in English
    except Exception as e:
        print(f"Failed to translate text '{text}': {e}")
        return text  # Return original text if translation fails


def calculate_relevancy(image_urls, subject, resize=False):
    """Calculate similarity scores between the subject and images."""
    scores = []
    subject_english = translate_to_english(subject)  # Ensure subject is in English

    for url in image_urls:
        image = fetch_image_resize(url) if resize else fetch_image(url)
        if image is not None:
            try:
                inputs = processor(
                    text=[subject_english],
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                scores.append(logits_per_image.item())
            except Exception as e:
                print(f"Error processing image from {url}: {e}")
    return scores


# Load the data
df = pd.read_excel(r"C:\Users\Sendsteps\Desktop\dataset1.xlsx", engine='openpyxl')
df['imgURLs'] = df['imgURLs'].apply(lambda x: x.split('||'))

# Calculate similarity scores with and without resizing
similarity_scores_resized = []
similarity_scores_original = []

for index, row in df.iterrows():
    print(f"Processing row {index}...")
    subj = row['subject']

    # Compute similarity scores
    scores_resized = calculate_relevancy(row['imgURLs'], subj, resize=True)
    scores_original = calculate_relevancy(row['imgURLs'], subj, resize=False)

    # Store in DataFrame
    similarity_scores_resized.append(scores_resized)
    similarity_scores_original.append(scores_original)

df['similarity-score-resized'] = similarity_scores_resized
df['similarity-score-original'] = similarity_scores_original

# Compute global mean and std across all scores
all_scores_resized = [score for row in similarity_scores_resized for score in row]
all_scores_original = [score for row in similarity_scores_original for score in row]

global_mean_resized = torch.mean(torch.tensor(all_scores_resized))
global_std_resized = torch.std(torch.tensor(all_scores_resized))

global_mean_original = torch.mean(torch.tensor(all_scores_original))
global_std_original = torch.std(torch.tensor(all_scores_original))

# Compute average scores per row
df['average_raw_score_resized'] = [torch.mean(torch.tensor(row)).item() if row else 0 for row in
                                   similarity_scores_resized]
df['average_raw_score_original'] = [torch.mean(torch.tensor(row)).item() if row else 0 for row in
                                    similarity_scores_original]

# Save the DataFrame to a new Excel file
output_path = r"C:\Users\Sendsteps\Desktop\scoring2.xlsx"
df.to_excel(output_path, index=False, engine='openpyxl')

print(f"Processed DataFrame has been exported to {output_path}")

print(f"Global Mean (Resized): {global_mean_resized.item():.4f}")
print(f"Global Std Dev (Resized): {global_std_resized.item():.4f}")

print(f"Global Mean (Original): {global_mean_original.item():.4f}")
print(f"Global Std Dev (Original): {global_std_original.item():.4f}")
