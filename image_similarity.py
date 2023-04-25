import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from keras.applications import vgg16
from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

class ImageSimilarity:
    def __init__(self, imgs_path_dict, nb_closest_images=10):
        self.nb_closest_images = nb_closest_images
        self.imgs_model_width, self.imgs_model_height = 224, 224

        self.vgg_model = vgg16.VGG16(weights='imagenet')

        self.sub_categories = {}
        for category, sub_categories in imgs_path_dict.items():
            for sub_category, path in sub_categories.items():
                feat_extractor = Model(inputs=self.vgg_model.input, outputs=self.vgg_model.get_layer("fc2").output)
                files = [path + "/" + x for x in os.listdir(path) if "jpg" in x]
                imported_images = self.load_images(files)
                imgs_features = self.extract_features(feat_extractor, imported_images)
                cos_similarities_df = self.calculate_cosine_similarity(files, imgs_features)

                self.sub_categories[f"{category}_{sub_category}"] = {
                    "path": path,
                    "feat_extractor": feat_extractor,
                    "files": files,
                    "imported_images": imported_images,
                    "imgs_features": imgs_features,
                    "cos_similarities_df": cos_similarities_df
                }

    def load_images(self, files):
        imported_images = []
        for f in files:
            original = load_img(f, target_size=(self.imgs_model_width, self.imgs_model_height))
            numpy_image = img_to_array(original)
            image_batch = np.expand_dims(numpy_image, axis=0)
            imported_images.append(image_batch)

        return np.vstack(imported_images)

    def extract_features(self, feat_extractor, imported_images):
        processed_imgs = preprocess_input(imported_images.copy())
        return feat_extractor.predict(processed_imgs)

    def calculate_cosine_similarity(self, files, imgs_features):
        cos_similarities = cosine_similarity(imgs_features)
        return pd.DataFrame(cos_similarities, columns=files, index=files)

    def save_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for sub_category, data in self.sub_categories.items():
            with open(f"{save_dir}/{sub_category}_feat_extractor.pkl", 'wb') as f:
                pickle.dump(data["feat_extractor"], f)


    def load_models(self, save_dir):
        for sub_category, data in self.sub_categories.items():
            with open(f"{save_dir}/{sub_category}_feat_extractor.pkl", 'rb') as f:
                data["feat_extractor"] = pickle.load(f)

    def load_feat_extractor(self, path):
        self.feat_extractor = load_model(path)

    def retrieve_most_similar_products(self, given_img, category, sub_category):
        key = f"{category}_{sub_category}"
        if key not in self.sub_categories:
            print(f"Category '{category}' and sub-category '{sub_category}' not found.")
            return

        data = self.sub_categories[key]

        print("Original product:")
        original = load_img(given_img, target_size=(self.imgs_model_width, self.imgs_model_height))
        plt.imshow(original)
        plt.show()

        print("Most similar products:")
        closest_imgs = data["cos_similarities_df"][given_img].sort_values(ascending=False)[1:self.nb_closest_images+1].index
        closest_imgs_scores = data["cos_similarities_df"][given_img].sort_values(ascending=False)[1:self.nb_closest_images+1]

        for i in range(len(closest_imgs)):
            original = load_img(closest_imgs[i], target_size=(self.imgs_model_width, self.imgs_model_height))
            plt.imshow(original)
            plt.show()
            print("Similarity score: ", closest_imgs_scores[i])

imgs_path_dict = {
    "Apparel": {
        "Boys": "/Users/rohittiwari/Desktop/vgg_image-net/data/Apparel/Boys",
        "Girls": "/Users/rohittiwari/Desktop/vgg_image-net/data/Apparel/Girls"
    },
    "Footwear": {
        "Boys": "/Users/rohittiwari/Desktop/vgg_image-net/data/Footwear/Men",
        "Girls": "/Users/rohittiwari/Desktop/vgg_image-net/data/Footwear/Women"
    }
}

image_similarity = ImageSimilarity(imgs_path_dict)
image_similarity.save_models("inference_models_pkl")

# image_similarity.load_models("inference_models_pkl")
# given_img = "/content/drive/MyDrive/data/Apparel/Boys/2714.jpg"
# image_similarity.retrieve_most_similar_products(given_img, "Apparel", "Boys")
      
