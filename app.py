from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import cv2
import os
import numpy as np
from mtcnn import MTCNN

# Initialize MTCNN face detector
detector = MTCNN()

# Initialize the VGGFace model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Rename layers to ensure unique names
for i, layer in enumerate(model.layers):
    layer._name = f"{layer.name}_{i}"

# Load precomputed feature embeddings and filenames
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

st.title('Which Bollywood Celebrity Are You?')

uploaded_image = st.file_uploader('Choose an image')

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path, model, detector):
    # Load the image from the given path
    img = cv2.imread(img_path)
    # Detect faces in the image
    results = detector.detect_faces(img)
    # Extract the bounding box coordinates
    x, y, width, height = results[0]['box']
    # Crop the face from the image
    face = img[y:y + height, x:x + width]
    # Resize the face image and convert it to an array
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    # Preprocess the face image for the model
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    # Extract features using the model
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features, top_n=5):
    similarity = []
    # Compute cosine similarity between the input features and all stored features
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), (feature_list[i]).reshape(1, -1))[0][0])
    # Get the indices of the top N most similar features
    top_indices = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[:top_n]
    return top_indices

if uploaded_image is not None:
    # Save the uploaded image in the 'uploads' directory
    if save_uploaded_image(uploaded_image):
        # Load the uploaded image for display
        display_image = Image.open(uploaded_image)
        # Extract features from the uploaded image
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
        # Get the top 5 recommended matches
        top_indices = recommend(feature_list, features, top_n=1)
        # Display the results
        col1, col2 = st.columns(2)

        from keras_vggface.utils import preprocess_input
        from keras_vggface.vggface import VGGFace
        import numpy as np
        import pickle
        from sklearn.metrics.pairwise import cosine_similarity
        import cv2
        from mtcnn import MTCNN
        from PIL import Image

        # as pickle.load(open('embedding.pkl','rb')) is in list format so converting it to array
        feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
        filenames = pickle.load(open('filenames.pkl', 'rb'))

        # now we will extract the features from image by passing it into the model
        model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

        detector = MTCNN()

        # load image -> face detection
        sample_img = cv2.imread('sample/ranbir.jpeg')

        results = detector.detect_faces(sample_img)

        x, y, width, height = results[0]['box']

        face = sample_img[y:y + height, x:x + width]

        # and extract its features
        image = Image.fromarray(face)
        image = image.resize((224, 224))
        face_array = np.asarray(image)

        face_array = face_array.astype('float32')
        expanded_img = np.expand_dims(face_array, axis=0)

        preprocessed_img = preprocess_input(expanded_img)
        result = model.predict(preprocessed_img).flatten()

        # then find the cosine distance of current image with all 1200s features
        similarity = []
        for i in range(len(feature_list)):
            similarity.append(cosine_similarity(result.reshape(1, -1), (feature_list[i]).reshape(1, -1))[0][0])

        index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]

        temp_img = cv2.imread(filenames[index_pos])
        cv2.imshow('output', temp_img)
        cv2.waitKey(0)
        with col1:
            st.header('Your Image')
            st.image(display_image)

        with col2:
            st.header("You look like:")
            for index, _ in top_indices:
                predicted_actor = " ".join(filenames[index].split('\\')[1].split('-'))
                st.write(predicted_actor)
                st.image(filenames[index], width=150)
