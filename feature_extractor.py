
# Import necessary libraries and modules
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

# Load filenames from a pickle file
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Initialize the VGGFace model with specific configurations
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


# Function to extract features from an image
def feature_extractor(img_path, model):
    # Load the image from the given path and resize it to 224x224 pixels
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert the image to a NumPy array with shape (224, 224, 3)
    img_array = image.img_to_array(img)

    # Expand dimensions to create a batch of size 1 with shape (1, 224, 224, 3)
    expanded_img = np.expand_dims(img_array, axis=0)

    # Preprocess the image to match the input format required by the VGGFace model
    preprocessed_img = preprocess_input(expanded_img)

    # Predict the features of the preprocessed image using the model
    result = model.predict(preprocessed_img).flatten()

    # Return the extracted features
    return result


features=[]

for file in tqdm(filenames):
    features.append(feature_extractor(file,model))


pickle.dump(features,open('embedding.pkl','wb'))