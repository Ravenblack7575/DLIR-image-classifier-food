#### DLIR-image-classifier-food
## SDGAI DLIR assignment - Identification of Food Item via Images Using Convolutional Neural Network (CNN) models

Generated summary of project using NotebookLM:

### About this project

This project focused on exploring and implementing deep learning models, specifically Convolutional Neural Networks (CNNs), for the classification of food images. The primary **aim** was to design and implement a CNN-based model that could accurately classify an image of a food dish into one of 10 specified food categories. The provided dataset consisted of 10 different food classes—including beet salad, hamburger, lasagna, and risotto—with a total of 7,500 images set aside for training, 2,000 for validation, and 500 as unseen data for testing. This classification task was noted as challenging because the same food dish can vary significantly in its arrangement, size, and color depending on preparation and presentation.


### Methods

The **methods** involved a two-pronged approach: first, developing and optimizing a custom-built CNN model, and second, utilizing transfer learning by integrating customized dense layers with a pre-trained convolutional base. Initial data processing involved normalizing and resizing the colored images to 150 x 150 pixels. To improve generalization and mitigate overfitting, extensive **data augmentation** was applied to the training images, utilizing techniques like rotation, width/height shifts, zooming, horizontal flips, and brightness adjustments. The custom-built model (Model 1-5A5) was iteratively adjusted, incorporating strategies such as adding Dropout and Batch Normalization layers, changing kernel sizes, and optimizing hyperparameters like the learning rate and optimizer, eventually settling on a five-block CNN architecture. For the transfer learning approach, the **InceptionV3** architecture was selected. Its pre-trained convolutional base was initially frozen, and only the custom dense layers were trained. Fine-tuning was then performed by unfreezing selected convolutional layers, starting from 'Mix 4,' to adapt the model to the specific dataset, and the Adam optimizer was used for the final training round.


### Results
The **results** clearly demonstrated the superiority of the transfer learning approach. The final optimized model, **Food\_IN6 (InceptionV3 convolutional base + customized Dense layers)**, achieved a high test accuracy of **0.8750** (87%), significantly outperforming the final custom-built model, Model 1-5A5, which recorded a test accuracy of 0.7659. The application of transfer learning was key in shortening the process while achieving better performance. The Food\_IN6 model was further validated by correctly classifying all four previously unseen food images taken from the internet with high prediction confidence scores. The model is available for testing via a demo hosted on Hugging Face Spaces.


=================================================================

### Demo

Demo of classifier is available at https://huggingface.co/spaces/Ravenblack7575/DLIRfood
