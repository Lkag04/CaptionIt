import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer# type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences# type: ignore
from tensorflow.keras.models import Model, load_model# type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Add, Bidirectional# type: ignore
from tensorflow.keras.applications import ResNet50# type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array# type: ignore
from tqdm import tqdm
import pickle
import json

# Load and preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
# Load captions dataset
def load_captions(json_path):
    with open(json_path, 'r') as f:
        captions = json.load(f)
    return captions

# Tokenize and prepare captions
def prepare_text_sequences(captions, max_words=5000, max_length=30):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")

    # Add <startseq> and <endseq>
    captions = [f"startseq {cap} endseq" for cap in captions]

    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)

    # Separate input and output (X: startseq -> word, Y: word -> endseq)
    X, Y = [], []
    for seq in sequences:
        for i in range(1, len(seq)):
            X.append(seq[:i])  # Input sequence
            Y.append(seq[i])   # Next word to predict

    X_padded = pad_sequences(X, maxlen=max_length, padding='post')
    return tokenizer, X_padded, np.array(Y)
# Extract features using ResNet50
def extract_features(image_dir, cache_file="image_features.pkl"):
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            print("Loading precomputed image features...")
            return pickle.load(f)
    print("Extracting image features...")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = {}
    for img_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        img = preprocess_image(img_path)
        feature = model.predict(img)[0]
        features[img_name] = feature

    with open(cache_file, "wb") as f:
        pickle.dump(features, f)

    return features

# Define CNN + LSTM model
#unet archetecute should be used self atention mechanism -bir
# after the relu activation func we should makel a layer with the swish activatio function by googgle. 
def build_model(vocab_size, max_length, embedding_dim=512, lstm_units=512):
    # Image Feature Input
    inputs_img = tf.keras.Input(shape=(2048,))
    # Change output dimension to 1024 to match the text branch output
    img_dense = Dense(1024, activation='relu')(inputs_img)
    img_dropout = Dropout(0.4)(img_dense)
    # Text Sequence Input
    inputs_txt = tf.keras.Input(shape=(max_length,))
    txt_embed = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs_txt)
    txt_lstm = LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(txt_embed)
    txt_lstm2 = LSTM(1024, dropout=0.3, recurrent_dropout=0.3)(txt_lstm)  # Increase from 256 → 1024
    #txt_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.3))(txt_embed)
    #txt_lstm2 = Bidirectional(LSTM(lstm_units, dropout=0.3))(txt_lstm)
    # Now both branches output shape (None, 1024)
    decoder = Add()([img_dropout, txt_lstm2])
    decoder_dense = Dense(vocab_size, activation='softmax')(decoder)

    model = Model(inputs=[inputs_img, inputs_txt], outputs=decoder_dense)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


# Load dataset
image_dir = "/Users/lakshyaagarwal/Downloads/career/projects/ai_captioner/flickr8k/Images"
captions_path = captions_path = "/Users/lakshyaagarwal/Downloads/career/projects/ai captioner/adityajn105/flickr8k/versions/1/captions.json"

captions_data = load_captions(captions_path)
tokenizer, X_texts, Y_labels = prepare_text_sequences(list(captions_data.values()))

# Load or extract image features
image_features = extract_features(image_dir)

# Convert images to training format
X_images = [image_features[img] for img in image_features.keys()]
X_images = np.array(X_images)

# Reduce dataset for quick testing (optional)
small_sample = min(5000, len(X_images))  # Use more samples if available
X_images, X_texts, Y_labels = X_images[:small_sample], X_texts[:small_sample], Y_labels[:small_sample]

# Build and train model
vocab_size = len(tokenizer.word_index) + 1
max_length = 30
########################
model = build_model(vocab_size, max_length)
model.fit([X_images, X_texts], Y_labels, batch_size=16, epochs=5, validation_split=0.2)
#########################
# Save model and tokenizer
model.save("image_captioning_model.keras")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Load model for inference
def load_trained_model():
    model = load_model("image_captioning_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Generate captions for an image
def generate_caption(image_path, model, tokenizer, max_length=30):
    img_name = os.path.basename(image_path)

    if img_name not in image_features:
        print(f"❌ Error: {img_name} not found in precomputed features!")
        return None

    image_feature = image_features[img_name]

    start_token = tokenizer.word_index.get("startseq")
    if start_token is None:
        print("❌ Error: 'startseq' token is missing from tokenizer vocabulary!")
        return None

    input_seq = [start_token]
    caption_generated = []

    for _ in range(max_length):
        sequence = pad_sequences([input_seq], maxlen=max_length, padding='post')
        y_pred = model.predict([np.array([image_feature]), np.array(sequence)])
        predicted_word_index = np.argmax(y_pred)
        predicted_word = tokenizer.index_word.get(predicted_word_index, '<unk>')

        if predicted_word == 'endseq':
            break

        caption_generated.append(predicted_word)
        input_seq.append(predicted_word_index)

    return ' '.join(caption_generated)

# Example usage
if __name__ == "__main__":
    model, tokenizer = load_trained_model()
    test_image = "/Users/lakshyaagarwal/Downloads/career/projects/ai_captioner/flickr8k/Images/33108590_d685bfe51c.jpg"

    caption = generate_caption(test_image, model, tokenizer)
    print("Generated Caption:", caption)
