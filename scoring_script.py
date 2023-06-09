import json
import os
import numpy as np
import pickle
import tensorflow
from azureml.core.model import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image

# This runs when the Docker container is started
def init():
    global model
    global feature_list
    global filenames
    
    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered
    model_dir = Model.get_model_path(model_name='resnet50_pickle_models')

    # Load the model from a file
    model = ResNet50(weights='imagenet',include_top=False,input_shape=(244,244,3))
    model.trainable = False
    model = tensorflow.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

    feature_list = np.array(pickle.load(open(os.path.join(model_dir, 'features_list_for_prods.pkl'),'rb')))
    filenames = pickle.load(open(os.path.join(model_dir, 'filenames_products.pkl'),'rb'))

# This runs when a request is made to the scoring API
def run(input_data):
    
    def feature_extraction(img_path, model):
        img = image.load_img(img_path, target_size=(244, 244))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result

    def recommend(features, feature_list, n_recommendations=8):
        neighbors = NearestNeighbors(n_neighbors=n_recommendations + 1, algorithm='brute', metric='cosine')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices, distances
    
    # Convert the input data into a PIL Image object
    input_image = Image.open(input_data)
    
    # Extract the features of the image
    features = feature_extraction(input_image, model)
    
    # Generate the recommendations
    indices, distances = recommend(features, feature_list)

    # Construct a response object
    result = {
        'indices': indices.tolist(),
        'distances': distances.tolist(),
    }

    return result
