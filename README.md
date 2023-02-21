# Ensemble Model for Liver Fat Classification using Pre trained CNNs

The provided code performs liver fat classification using a deep learning ensemble model. The code starts with importing necessary libraries, including numpy, keras, tensorflow, PIL, sklearn, and others.

The data is loaded from the original dataset directory and is divided into four groups based on the filenames. A function is defined to load and resize the images to a specific target size. The pre-trained models ResNet50, VGG16, and InceptionV3 are loaded and their layers are frozen.

The ensemble model is then created by extracting features from the three pre-trained models, combining them, and training a dense layer on top of them using the softmax activation function. The ensemble model is compiled using the Adam optimizer and categorical cross-entropy loss.

Finally, the ensemble model is trained using a data generator and evaluated using the test generator. The accuracy score is computed using the true and predicted classes. The code provides an example of how to create an ensemble model using pre-trained models for image classification tasks.
