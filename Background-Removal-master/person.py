from keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2  # Optional, for future image processing needs

# Load the pre-trained model
# Provide the correct path to your model file (e.g., 'main_model.hdf5')
model = load_model('main_model.hdf5', compile=False)

graph = tf.get_default_graph()


def predict(image):
    """Makes a prediction on an image using the loaded model."""
    with graph.as_default():
        # Make prediction
        prediction = model.predict(image[None, :, :, :])
        prediction = prediction.reshape((224, 224, -1))
    return prediction


def remove_background_and_predict(image_path):
    """
    Removes background using PIL's `convert` method and makes a prediction.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image: The image with the predicted mask applied as transparency.
    """
    try:
        image = Image.open(image_path).convert('RGBA')  # Open as RGBA for transparency
        image = image.resize((224, 224))  # Resize for prediction

        prediction = predict(np.array(image)[:, :, 0:3] / 255.0)
        prediction = prediction[:, :, 1].reshape((image.height, image.width))

        # Process prediction for transparency mask (adjust thresholds as needed)
        mask = np.where(prediction > 0.5, 255, 0)

        # Combine original image with mask (replace with your desired blending method)
        transparency = Image.alpha_composite(image, image.copy().convert('L').point(lambda p: p if p > mask.max() else 0))

        transparency.save('output.png')
        print("Saved the output image in output.png")
        return transparency

    except FileNotFoundError:
        print("Error: File not found at", image_path)
        return None


def main():
    """Prompts user for image path, performs background removal and prediction."""
    print('##################################################################')
    print()
    path = input('Enter path of file: ')
    print()
    print('##################################################################')
    print()

    processed_image = remove_background_and_predict(path)

    if processed_image:
        print('Prediction complete. Transparency applied to the output image.')
    print('##################################################################')


if __name__ == '__main__':
    print("succ")
    main()