import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained Keras model
model = tf.keras.models.load_model('food_IN6.keras')

img_size = 150 # Assuming your model expects 150x150 images
class_names = ['beet_salad', 'beignets', 'eggs_benedict', 'hamburger', 'hot_and_sour_soup', 'huevos_rancheros', 'lasagna', 'risotto', 'seaweed_salad', 'strawberry_shortcake'] # Your actual class names

def predict_image(image):
    # Preprocess the image for your model
    img = Image.fromarray(image).resize((img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Rescale if your model was trained with 1./255

    predictions = model.predict(img_array)[0]
    confidence = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

    # Check if the highest prediction is below the threshold
    max_confidence = max(confidence.values())
    if max_confidence < 0.85:  # 85% threshold
        return "Classification not possible."
    else:
        return confidence

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload Food Image"),
    outputs=gr.Label(num_top_classes=len(class_names) if True else None), # Modified to handle string output
    title="My 'Awesome' Food Classifier",
    description="This classifier was done as part of a course assignment. It is food classifier trained to classify images into one of 10 specific food classes: 1) beet salad, 2) beignets, 3) eggs benedict, 4) hamburger, 5) hot and sour soup, 6) huevos rancheros, 7) lasagna, 8) risotto, 9) sea-weed salad and 10) strawberry shortcake. If the food image uploaded can't be confidently match to a food class (<85%), it should report 'Classification not possible.' Upload an image and try it out!."
)

iface.launch()