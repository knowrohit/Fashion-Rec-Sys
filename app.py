from colorthief import ColorThief  # Add this line
import matplotlib.pyplot as plt  
from gtts import gTTS
from tempfile import NamedTemporaryFile
import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm




feature_list = np.array(pickle.load(open('features_list_for_prods.pkl','rb')))
filenames = pickle.load(open('filenames_products.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(244,244,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.set_page_config(
    page_title="FashionGPT",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('FashionGPT')



page_route = ["Home", "About"]
choice = st.sidebar.selectbox("Select Activity", page_route)
st.sidebar.markdown(
        """ Developed by an ML enthusiast himself:
        
        Rohit Tiwari 
        Email : knowrohit.07@gmail.com
            """)

quotes = [
    "“Do I really look like a guy with a plan? You know what I am? I'm a dog chasing cars. I wouldn't know what to do with one if I caught it! You know, I just... *do* things.” — Joker",
    "“Why so serious?” — Joker",
    "“Introduce a little anarchy. Upset the established order, and everything becomes chaos. I'm an agent of chaos...” — Joker",
    "“You see, madness, as you know, is like gravity. All it takes is a little push!” — Joker",
]


quote = st.sidebar.selectbox("Feedback From Beta Testers ", quotes)

if choice == "Home":
        html_temp_home1 = """<div style="background-color:#0a2342;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Product recommender system using Transfer learning and Unsupervised learning.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

elif choice == "About":
        st.subheader("About FashionGPT")
        html_temp_about1= """<div style="background-color:#0a2342;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    FashionGPT: A Product Recommender System using Transfer Learning</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        st.write("FashionGPT is an AI-powered product recommender system designed to help users find their perfect fashion match. By leveraging the power of Transfer Learning, FashionGPT can analyze the visual features of a given image and recommend similar products that cater to the user's unique style preferences.")
        
        st.write("## How it Works")
        st.write("FashionGPT uses a pre-trained ResNet50 model as a feature extractor to analyze the visual aspects of a user's selected image. It then compares the extracted features with a database of product images to find the most visually similar items. The Nearest Neighbors algorithm is employed to identify the top 5 matching products based on their similarity scores.")
        
        st.write("## Key Features")
        st.write("- State-of-the-art deep learning model for feature extraction")
        st.write("- Efficient and accurate product recommendations")
        st.write("- User-friendly interface for seamless interaction")
        st.write("- Wide range of supported fashion products")
        st.write("- Continually updated and refined to deliver the best results")
        
        st.write("## Team")
        st.write("FashionGPT is developed by a dedicated AI enthusiast and fashion aficionado, who believe in harnessing the power of technology to improve the shopping experience for users. He is committed to delivering innovative solutions that help users find their ideal fashion products with ease and accuracy.")
        
        st.write("## Contact Us")
        st.write("We would love to hear from you! If you have any questions, suggestions, or feedback, please feel free to reach out to us at [knowrohit.work@gmail.com](mailto:knowrohit.work@gmail.com). You can also connect with us on [Twitter](https://twitter.com/knowrohit07).")
        
        st.write("## Acknowledgements")
        st.write("I would like to express our gratitude to the following resources and organizations for their invaluable support and contributions to the development of FashionGPT:")
        st.write("- [Rahul Tiwari](https://twitter.com/rahul_tiwari95) for the inspiration lmao")
        st.write("- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework")
        st.write("- [Streamlit](https://streamlit.io/) for the web application framework")
        st.write("- [ResNet50](https://arxiv.org/abs/1512.03385) for the pre-trained deep learning model")
        
        st.write("## Disclaimer")
        st.sidebar.markdown("## Connect with me")
        st.sidebar.markdown(
    """<a href="https://twitter.com/knowrohit07" target="_blank"><img src="https://img.shields.io/badge/Twitter--_.svg?style=social&logo=twitter" alt="Rohit Tiwari Twitter"></a>
    """, unsafe_allow_html=True)


        # st.markdown(html_temp4, unsafe_allow_html=True)

else:
   pass




def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(244, 244))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def display_color_palette(img_path, num_colors=5):
    color_thief = ColorThief(img_path)
    palette = color_thief.get_palette(color_count=num_colors)
    plt.figure(figsize=(5, 1))
    plt.bar(range(num_colors), [1] * num_colors, color=[f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in palette], width=1)
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.close()

# def recommend(features,feature_list):
#     neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
#     neighbors.fit(feature_list)

#     distances, indices = neighbors.kneighbors([features])

#     return indices, distances

def recommend(features, feature_list, n_recommendations=8):  # Modify this line
    neighbors = NearestNeighbors(n_neighbors=n_recommendations + 1, algorithm='brute', metric='cosine')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices, distances



if not os.path.exists('uploads'):
    os.makedirs('uploads')
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        show_original_image = st.checkbox('Show original image alongside recommendations')  # Add this line
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        # st.text(features)
        # recommendention
        # indices, distances = recommend(features,feature_list)
        number_of_recommendations = st.slider('Number of recommendations:', min_value=1, max_value=10, value=5, step=1)
        indices, distances = recommend(features, feature_list, number_of_recommendations)  # Modify this line

        show_stats = st.button("STATS FOR NERDS")

        number_of_recommendations = 8  # Adjust this value as needed
        columns = st.columns(1 + show_original_image * number_of_recommendations)
        for i in range(number_of_recommendations):
            if i == 0 and show_original_image:
                columns[i].image(display_image)
            else:
                image_index = i - 1 if show_original_image else i
                # Check if image_index + 1 is within the range of indices
                if image_index + 1 < len(indices[0]):
                    image_path = filenames[indices[0][image_index + 1]]
                    columns[i + show_original_image].image(Image.open(image_path))
                else:
                    st.warning(f"The magic lies within the numbers.")


        if show_stats:
            st.write("Detailed information for the recommended products:")
            for i, distance in enumerate(distances[0][1:1 + number_of_recommendations]):  # Modify this line
                with st.expander(f"Product {i + 1}"):
                    st.write(f"Product {i + 1}:")
                    st.write(f"Similarity score: {1 - distance:.4f}")
                    st.write(f"Filename: {filenames[indices[0][i]]}")
                    
                    img = Image.open(filenames[indices[0][i]])
                    img_dimensions = img.size
                    st.write(f"Image dimensions: {img_dimensions}")
                    
                    aspect_ratio = img_dimensions[0] / img_dimensions[1]
                    st.write(f"Aspect ratio: {aspect_ratio:.2f}")

                    st.write(f"Index in feature list: {indices[0][i]}")
                    st.write(f"Raw distance score: {distance:.4f}")
                    st.write("Color palette:")
                    display_color_palette(filenames[indices[0][i]])
                    st.write("")
    else:
        st.header("Some error occured in file upload")
