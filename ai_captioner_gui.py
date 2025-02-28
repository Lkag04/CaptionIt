import os
import numpy as np
import tensorflow as tf
import pickle
import json
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.sequence import pad_sequences# type: ignore
from tensorflow.keras.models import load_model# type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input# type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array# type: ignore


# Load and preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Extract features from a new image using ResNet50
def extract_feature_from_image(image_path):
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    img = preprocess_image(image_path)
    feature = model.predict(img)[0]
    return feature


# Generate caption for the uploaded image
def generate_caption(image_path, model, tokenizer, max_length=30):
    image_feature = extract_feature_from_image(image_path)

    start_token = tokenizer.word_index.get("startseq", 1)
    input_seq = [start_token]

    for _ in range(max_length):
        sequence = pad_sequences([input_seq], maxlen=max_length, padding="post")
        y_pred = model.predict([np.array([image_feature]), np.array(sequence)])
        predicted_word_index = np.argmax(y_pred)
        predicted_word = tokenizer.index_word.get(predicted_word_index, "<unk>")

        if predicted_word == "endseq":
            break

        input_seq.append(predicted_word_index)

    caption = " ".join([tokenizer.index_word.get(idx, "<unk>") for idx in input_seq])
    return caption.replace("startseq", "").replace("endseq", "").strip()


# Load model and tokenizer
print("🔄 Loading model and tokenizer...")
model_path = "image_captioning_model.keras"
tokenizer_path = "tokenizer.pkl"

model = load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename(
    filetypes=[("JPEG Files", "*.jpg *.jpeg"), ("PNG Files", "*.png"), ("All Files", "*.*")]
)

    
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize for display
        img = ImageTk.PhotoImage(img)

        image_label.config(image=img)
        image_label.image = img  # Keep reference

        caption = generate_caption(file_path, model, tokenizer)
        caption_label.config(text=f"Generated Caption:\n{caption}")




# Create GUI window
root = tk.Tk()
root.title("AI Image Captioner")
root.geometry("500x600")

# Welcome message
welcome_label = Label(root, text="📸 AI Image Captioner", font=("Arial", 16, "bold"))
welcome_label.pack(pady=10)

# Upload button
upload_btn = Button(root, text="📂 Upload Image", command=upload_image, font=("Arial", 12), bg="#4CAF50", fg="white")
upload_btn.pack(pady=10)

# Image display area
image_label = Label(root)
image_label.pack(pady=10)

# Caption display area
caption_label = Label(root, text="", font=("Arial", 14), wraplength=400, justify="center")
caption_label.pack(pady=10)

# Run the GUI event loop
root.mainloop()
