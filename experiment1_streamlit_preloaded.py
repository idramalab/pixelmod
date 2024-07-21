#List of Imports 
import streamlit as st
import subprocess
import requests
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusException
import logging
import pandas as pd
import json
import glob
import uuid
import pdqhash
from PIL import Image
import io
import pymilvus as milvus
import cv2
from pymilvus import Collection
import os

# Setup logging
logging.basicConfig(level=logging.INFO)


# Clear cache button
if st.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared. Please refresh the page.")



# Milvus parameters
#Configuring setup for Milvus collection 
HOST ='standalone'#Set this as standalone for testing within an internal setup, or 127.0.0.1 for testing locally
PORT = '19530'
TOPK = 10
default_dim = 256  # dimension of embedding extracted by MODEL
COLLECTION_NAME = 'usenixtest_pdqhash_v2' #Name of collection to be used 
INDEX_TYPE = 'BIN_FLAT'
METRIC_TYPE = 'HAMMING'
default_vec_field_name='pdqhash'#Field used for vector embedding while indexing the collection


#List of Files for OCR metadata . For Both source as well as index images
import json
with open('final_manipulation_ocr_en.json') as f:
    manipulation_image_ocr_en = json.load(f)
with open('final_source_ocr_en.json') as f:
    source_image_ocr_en = json.load(f)


# Function to compute PDQHash for an image
@st.cache_data(show_spinner=False)

def pdqhash_func(image_path):
    try:
        # Load image using PIL
        # img = Image.open(image_path).convert('RGB')
        img = Image.open(io.BytesIO(image_path)).convert('RGB')

        # Convert image to numpy array
        img_array = np.array(img)

        # Compute PDQHash
        hash_vector, _ = pdqhash.compute(img_array) # Placeholder for actual PDQHash computation

        if len(hash_vector) != 256:
            print("Error: Hash vector does not have length of 256")
            return None
        
        # Convert binary vector to bytes using np.packbits
        bytes_data = bytes(np.packbits(hash_vector))

        return hash_vector,bytes_data

    except Exception as e:
        print(f"Error computing PDQHash for image '{image_path}': {e}")
        return b''

# Define a function to generate binary embeddings from hash values
def generate_embedding_from_hash(hash_value):
    try:
        # Pad hash to ensure it is 64 characters long
        padded_hash = pad_pdqhash(hash_value)
        # Convert the hash value to a binary embedding
        binary_embedding = hex_to_hash_pdq(padded_hash)
        
        # Check if binary_embedding is of bytes or bytearray type
        if not isinstance(binary_embedding, (bytes, bytearray)):
            raise ValueError("Binary embedding must be of bytes or bytearray type")
        
        # Check if binary_embedding has the correct length
        expected_length = default_dim // 8  # DIM is the expected dimension of the binary vector in bits
        if len(binary_embedding) != expected_length:
            raise ValueError(f"Binary embedding has incorrect length. Expected: {expected_length}, Actual: {len(binary_embedding)}")
        
        # Print out the binary embedding for debugging
        logging.info(f"Binary Embedding: {binary_embedding}")
        
        return binary_embedding
    
    except ValueError as e:
        logging.error(f"Error generating binary embedding from hash: {e}")
        return None

def kgram_similarity(s1, s2, k=4):
    if len(s1)==0 and len(s2)==0:
        return 1
    if len(s1)==0 or len(s2)==0:#Either of them empty means no intersection
        return 0
    s1_kgrams = set([s1[i:i+k] for i in range(len(s1)-k+1)])
    s2_kgrams = set([s2[i:i+k] for i in range(len(s2)-k+1)])
    intersection = len(s1_kgrams.intersection(s2_kgrams))
    union = len(s1_kgrams.union(s2_kgrams))
    return intersection/union




# Check hex string size
def check_hex_string_size(hexstr, expected_size):
    actual_size = len(hexstr)
    if actual_size != expected_size:
        raise ValueError(f"Expected hex string size of {expected_size}, but got size {actual_size}")

# Pad PDQ hash
def pad_pdqhash(image_pdqhash):
    if len(image_pdqhash) == 64:
        return image_pdqhash
    else:
        fill = 64 - len(image_pdqhash)
        fillstr = '0' * fill
        return fillstr + image_pdqhash



def hex_to_hash_pdq(hexstr, hash_size=16):
    count = hash_size * hash_size // 4
    if len(hexstr) != count:
        raise ValueError(f'Expected hex string size of {count}.')

    binary = bin(int(hexstr, 16))[2:].zfill(count * 4)
    val = np.array([int(b) for b in binary])

    return bytes(np.packbits(val, axis=-1).tolist())  # Returns a numpy array of 0s and 1s


# Create Milvus collection
def create_milvus_collection(collection_withpath, dim):
    #if utility.has_collection(collection_withpath):
    #    utility.drop_collection(collection_withpath)
    #    logging.info(f"Dropped existing collection '{collection_withpath}'")
    embedding_fields = [
        milvus.FieldSchema(name='uuid', dtype=milvus.DataType.VARCHAR, description='unique identifier', max_length=36,
                    is_primary=True, auto_id=True),
        milvus.FieldSchema(name=default_vec_field_name, dtype=milvus.DataType.BINARY_VECTOR, dim=default_dim),
        milvus.FieldSchema(name="imgname",dtype=milvus.DataType.VARCHAR,description = 'file path', max_length =500)
    ]
    default_schema = milvus.CollectionSchema(fields=embedding_fields, description="artifact perf eval")

    collection = milvus.Collection(collection_withpath,schema=default_schema)

    return collection

    

# Check if collection is loaded
def check_collection_loaded(collection_test):
    try:
        collection3 = Collection(collection_test)
        collection.load()
        st.success(f"Collection '{collection_test}' loaded successfully.")
        return collection3
    except MilvusException as e:
        st.error(f"Error loading collection '{collection_test}': {e}")
        return None

# Connect to Milvus
def connect_to_milvus(host, port):
    try:
        client=milvus.connections.connect(host=host, port=port)
        st.success("Connected to Milvus successfully.")
    except Exception as e:
        st.error(f"Error connecting to Milvus: {e}")
        st.stop()


# Connect to Milvus and create/load collection
connect_to_milvus(HOST, PORT)
collection = create_milvus_collection(COLLECTION_NAME, default_dim)


# Perform search in Milvus
def search_in_milvus(collection, query_embedding):
    logging.info(f"Search Radius inside fxn {radius}")
    try:
        search_params = {"metric_type": METRIC_TYPE, "params": {"radius": radius}}
        results = collection.search(
            data=[query_embedding],
            anns_field=default_vec_field_name,
            param=search_params,
            limit=collection.num_entities,
            output_fields=["imgname"]
        )

        logging.info(f"Search embeding : {query_embedding}")
        # Debugging: Print raw results
        logging.info(f"Raw search results: {results}")

        logging.info(f"Count : {collection.num_entities}")
        # Return the search results
        return results

    except Exception as e:
        st.error(f"Error during search: {e}")
        return None

def display_search_results(results,imgname,mode):
    if results is not None:
        st.write("Search Results without applying OCR:")
        images=[]
        for result in results[0]:
            #image_url = result.entity.get('url')
            image_file = result.entity.get('imgname')
            raw_folder_path="ManipulationTarget/"
            if os.path.isfile(raw_folder_path+image_file+".jpg"):
                path=raw_folder_path+image_file+".jpg"
            elif os.path.isfile(raw_folder_path+image_file+".png"):
                path=raw_folder_path+image_file+".png"
            else:
                st.write("Neither jpg nor png exists")
            if image_file:
                 try:
                     with open(path, 'rb') as f:
                         file_image = f.read()
                     images.append(file_image)
                     #st.image(file_image, caption='File Image')
                 except FileNotFoundError as e:
                     logging.error(f"File not found at path: {image_file} - {e}")
                     st.warning(f"File not found at path: {image_file}")
                 except Exception as e:
                     logging.error(f"Error opening file at path: {image_file} - {e}")
                     st.warning(f"Error opening file at path: {image_file} - {e}")
            else:
                 st.warning("No file path found")
    logging.info(f"Images count {len(images)}, whereas results count is {len(results[0])}")
    st.session_state.rawimgs=images
    st.image(st.session_state.rawimgs,width=200)
    ####
    st.divider()
    st.divider()
    filtered_images=[]
    if results is not None:
        st.write("Now showing filtered results after applying OCR:")
        #Get this from image automatically
        sourcelabel=source_image_ocr_en[imgname].replace("\n"," ").lower()[:80]
        for result in results[0]:
            image_file = result.entity.get('imgname')
            mylabel=manipulation_image_ocr_en[image_file].replace("\n"," ").lower()[:80]
            if kgram_similarity(mylabel,sourcelabel,4)>=0.05:
                raw_folder_path="ManipulationTarget/"
                if os.path.isfile(raw_folder_path+image_file+".jpg"):
                    path=raw_folder_path+image_file+".jpg"
                elif os.path.isfile(raw_folder_path+image_file+".png"):
                    path=raw_folder_path+image_file+".png"
                else:
                    st.write("Neither jpg nor png exists")
                with open(path,'rb') as f:
                    filtered_images.append(f.read())
        logging.info(f"Filtered images count {len(filtered_images)}")
        st.session_state.ocrimgs=filtered_images
        st.image(st.session_state.ocrimgs,width=200)



if not collection:
    st.stop()

collection_Loaded = check_collection_loaded(COLLECTION_NAME)
if not collection_Loaded: #changed this
    st.stop()

st.header("Try some of the most popular images from VoterFraud dataset")
st.write("Click one of these images to search for results")


from st_clickable_images import clickable_images

import base64

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#Specify the list of preloaded images that you want user to click on 

preloaded_titles=["img21","img39","img98","img5","img2","img11","img19","img26","img53","img90"]
preloaded_imgfiles=["img21.jpg","img39.jpg","img98.jpg","img5.jpg","img2.jpg","img11.jpg","img19.jpg","img26.jpg","img53.jpg","img90.jpg"]
def trigger(clicked_index, image_title):
    st.write(f"Image #{image_title} (index {clicked_index}) was clicked.")
    with open("ManipulationSource/"+preloaded_imgfiles[clicked_index],"rb") as f:
        data=f.read()
    search_preloaded(data,image_title)
            
# Initialize session state for previous clicked value
if 'previous_clicked' not in st.session_state:
    st.session_state.previous_clicked = -1

# since uploaded image is not part of
#Trigger search from clicking of preloaded image
def search_preloaded(image_content,imagename):
    hash_str,bytes_data = pdqhash_func(image_content)
    if bytes_data is not None and len(bytes_data) > 0:
        st.session_state.bytes_data=bytes_data
        st.session_state.imagename=imagename
        image_search_results = search_in_milvus(collection, bytes_data)
        logging.info(f"Image source {imagename} and OCR label {source_image_ocr_en[imagename]}")
        display_search_results(image_search_results,imagename,"imageselected")
    else:
        st.warning("PDQ hash computation returned empty result.")


clicked = clickable_images([
        f"data:image/jpg;base64,{get_img_as_base64('ManipulationSource/'+img)}" for img in preloaded_imgfiles],
    titles=preloaded_titles,
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "200px"},
)


radius=91 #Threshold 90+1 for Milvus range search

st.session_state.rawimgs=[]
st.session_state.ocrimgs=[]

# Check if clicked value has changed
if clicked != st.session_state.previous_clicked:
    if clicked > -1:
        trigger(clicked, preloaded_titles[clicked])
    st.session_state.previous_clicked = clicked


