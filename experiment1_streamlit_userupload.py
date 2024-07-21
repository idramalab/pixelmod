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
HOST = 'standalone'##Set this as standalone for testing within an internal setup, or 127.0.0.1 for testing locally
PORT = '19530'
TOPK = 10
default_dim = 256  # dimension of embedding extracted by MODEL
COLLECTION_NAME = 'usenixtest_pdqhash_v2'
INDEX_TYPE = 'BIN_FLAT'
METRIC_TYPE = 'HAMMING'
default_vec_field_name='pdqhash'


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

def display_search_results(results,imgname):
    if results is not None:
        st.write("Output PixelMod with user uploaded image!!")
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


if not collection:
    st.stop()

collection_Loaded = check_collection_loaded(COLLECTION_NAME)
if not collection_Loaded: #changed this
    st.stop()

# Add a section for hash input and search
st.header("Search with PDQ Hash")
hash_input = st.text_input("Enter Hash (up to 64 characters):")

image_path = None
# Upload image for searching
uploaded_file = st.file_uploader("Upload an image for searching", type=["jpg", "jpeg", "png"])


radius=91 #Threshold 90+1 for Milvus range search

if st.button("Search with Uploaded Image"):
    # st.session_state.hash_input = ""  # Clear the hash input
    if uploaded_file is not None: 
        try: 
            image_content= uploaded_file.read()
            hash_str,bytes_data = pdqhash_func(image_content)
            logging.info(f"Computed hash bytes: {bytes_data}")
            if bytes_data is not None and len(bytes_data) > 0: 
                image_search_results = search_in_milvus(collection, bytes_data)
                display_search_results(image_search_results,"")
            else:
                st.warning("PDQ hash computation returned empty result.")

        except ValueError as e:
            st.error(f"Error converting image to binary vector: {e}")
            st.stop()
    else: 
        st.warning("Please upload an image before performing the search")

# Perform a search on user query
if st.button("Search with Hash"):
    if hash_input:
        # Step 1: Ensure that the input hash string is trimmed
        hash_input = hash_input.strip()

        # Step 2: Confirm that the input hash string is always exactly 64 characters long
        if len(hash_input) != 64:
            logging.info(f"Hash length :{len(hash_input)}")
            st.error("Error: Input hash string must be exactly 64 characters long.")
            st.stop()

        # Step 3: Print the input hash string before and after stripping whitespace
        print("Input hash string:", hash_input)

        try:
            binary_vector = hex_to_hash_pdq(pad_pdqhash(hash_input))

            # Step 4: Verify that the functions pad_pdqhash() and hex_to_hash_pdq() are working correctly
            print("Padded hash:", pad_pdqhash(hash_input))
            print("Binary vector:", binary_vector)

            hash_search_results = search_in_milvus(collection, binary_vector)
            print("Search results:", hash_search_results)
            display_search_results(hash_search_results,"")
        except ValueError as e:
            st.error(f"Error converting hash to binary vector: {e}")
            st.stop()
    else:
        st.warning("Please enter a hash value for searching.")

hash_search_results=None
image_search_results=None
# Display search results
if hash_search_results is not None:
    st.header("Search Results from Hash:")
    display_search_results(hash_search_results,"")

if image_search_results is not None:
    st.header("Search Results from Uploaded Image:")
    display_search_results(image_search_results,"")
