# CaptionIt
This project automatically generates descriptive captions for images by combining computer vision and natural language processing techniques. The system leverages deep learning to interpret the visual content of an image and then produce a coherent, natural language description.

# AI Image Captioner 🖼️📝

An AI-powered application that generates descriptive captions for images using a deep learning model combining ResNet50 for image feature extraction and LSTM for sequence prediction. Includes both a training script and a user-friendly GUI.

## Features ✨
- **ResNet50**: Extracts rich image features from input images.
- **LSTM Network**: Generates captions by learning from sequential text data.
- **GUI Interface**: Built with Tkinter for easy image upload and caption visualization.
- **Preprocessing Pipeline**: Handles image normalization and caption tokenization.
- **Kaggle Integration**: Script to download the Flickr8k dataset directly via Kaggle.

## Installation 🛠️

### Dependencies
- Python 3.8+
- Install required packages:
  bash
  pip install tensorflow keras numpy pillow tqdm kagglehub tkinter


### Dataset Setup
1. **Download the Flickr8k Dataset**:
   - Run `import dataset.py` to download the dataset via Kaggle.
   - Update the `image_dir` and `captions_path` in `ai_captioner.py` to point to your dataset location.

## Usage 🚀

### Training the Model
1. Run the training script:
    bash
   python ai_captioner.py
    
   - By default, it uses a small sample (5000 images) for quick testing. Adjust `small_sample` in the code for full training.

### Using the GUI
1. Run the GUI script:
    bash
   python ai_captioner_gui.py
    
2. Click **Upload Image** to select an image (JPEG/PNG).
3. View the generated caption below the image.

(demo_gui_screenshot.png) *(Replace with actual screenshot of your GUI in action)*

## File Structure 📂
- `ai_captioner.py`: Main training script (model definition, training, and caption generation logic).
- `ai_captioner_gui.py`: GUI application for image upload and caption display.
- `import dataset.py`: Downloads the Flickr8k dataset from Kaggle.

## Model Architecture 🧠
- **Image Branch**: ResNet50 (pre-trained on ImageNet) → Dense (1024 units) → Dropout.
- **Text Branch**: Embedding → LSTM (256 units) → LSTM (1024 units).
- **Decoder**: Combined features → Dense layer with softmax for word prediction.


## Acknowledgments 🙏
- Dataset: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Libraries: TensorFlow, Keras, and KaggleHub.
- Inspired by research in image captioning and attention mechanisms.
