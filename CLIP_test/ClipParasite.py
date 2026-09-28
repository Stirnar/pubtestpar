import torch
from PIL import Image
import transformers
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from google.cloud import storage
from io import BytesIO
client = storage.Client(project="parasites-495314")
bucket = client.bucket("stool-parasite-images")
dataset_path = r"G:\stool-parasite-images\raw\Lab microscope"

# Load BiomedCLIP from Hugging Face
from open_clip import create_model_from_pretrained, get_tokenizer
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
# ... pre-process image and tokenize labels, then compare using model ...

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

model.eval()

# Image Embedder (Local files)
def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(inputs)
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim = True)
    
    return image_feature.squeeze().cpu().numpy()

# Image Embedder (GCS files)
def embed_image_from_gcs(bucket_name, blob_name):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    image_bytes = blob.download_as_bytes()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(inputs)
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    
    return image_feature.squeeze().cpu().numpy()

# Class Embedder, but calculate/embed centroid for each class (Local files)
def embed_class(dataset_path):
    final_embedding = {}
    
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        embeddings = []
        
        for img in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img)
            emb = embed_image(img_path)
            embeddings.append(emb)
        embeddings = np.array(embeddings)
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        final_embedding[class_name] = centroid
    return final_embedding

def embed_class_from_gcs(bucket_name, dataset_path):
    final_embedding = {}
    
    blobs = bucket.list_blobs(prefix=dataset_path)
    class_images = {}
    
    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        class_name = blob.name.split('/')[-2]
        if class_name not in class_images:
            class_images[class_name] = []
        class_images[class_name].append(blob.name)
    
    for class_name, image_paths in class_images.items():
        embeddings = []
        
        for img_path in image_paths:
            emb = embed_image_from_gcs(bucket_name, img_path)
            embeddings.append(emb)
        embeddings = np.array(embeddings)
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        final_embedding[class_name] = centroid
    return final_embedding


# Predict function
def predict(image_path, final_embed):
    query_img = embed_image(image_path)
    similarity_scores = {}
    
    for class_name, class_emb in final_embed.items():
        score = cosine_similarity([query_img],[class_emb])[0][0]
        similarity_scores[class_name] = score
    final_scores = sorted(similarity_scores.items(), key = lambda x:x[1], reverse = True)
    return final_scores

            
def run_clip_local():
    class_embeddings = embed_class(dataset_path = dataset_path)
    class_embeddings_path = "parasite_embeddings1.pkl"
    with open(class_embeddings_path, "wb") as f:
        pickle.dump(class_embeddings, f)
    i = 0
    single_test_image = "C:/AI_Blood_Smear_Research/NewTest_Images/TestTime/leishmaniatest.png"
    test_images_path = "C:/AI_Blood_Smear_Research/NewTest_Images/TestTime"
    for im in os.listdir(test_images_path):
        test_image = os.path.join(test_images_path, im)
        with open(class_embeddings_path, "rb") as f:
            class_embeddings = pickle.load(f)
        
        result = predict(image_path=test_image, final_embed=class_embeddings)
        
        # print(result)
        print(f"Image {i}:")
        for parasite, score in result[:3]:
            print(parasite, score)
        print()
        i+=1
    
def run_clip_gcs():
    class_embeddings = embed_class_from_gcs(bucket_name="stool-parasite-images", dataset_path="raw/Lab microscope")
    class_embeddings_path = "parasite_embeddings_gcs.pkl"
    with open(class_embeddings_path, "wb") as f:
        pickle.dump(class_embeddings, f)
    i = 0
    test_images_path = "raw/TestTime"
    blobs = bucket.list_blobs(prefix=test_images_path)
    
    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        test_image = blob.name
        with open(class_embeddings_path, "rb") as f:
            class_embeddings = pickle.load(f)
        
        result = predict(image_path=test_image, final_embed=class_embeddings)
        
        # print(result)
        print(f"Image {i}:")
        for parasite, score in result[:3]:
            print(parasite, score)
        print()
        i+=1
