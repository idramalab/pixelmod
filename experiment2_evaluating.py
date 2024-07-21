#Loading Milvus library and OpenCV 
import pymilvus as milvus
import cv2
from pymilvus import Collection
import pdqhash
import numpy as np
from PIL import Image
import io
#Configuring setup for Milvus collection 
hostname="standalone"#Set this as standalone for testing within an internal setup, or 127.0.0.1 for testing locally
client=milvus.connections.connect(host=hostname, port=19530)

#Images used for building GTViz is indexed as collection usenixtest_pdqhash_v2 inside the Docker container. Replace the collection name with other variable for testing other collections.
collection_name="usenixtest_pdqhash_v2"
#Default dimension of 256 for PDQHash
default_dim=256
#Vector field used during indexing process. Make sure it matches the same field specified during creating the collection 
default_vec_field_name="pdqhash"

#Re-using the same Collection Schema used during creating the index. Schema contains unique Id, image raw path, and vector embedding for the images
embedding_fields = [
        milvus.FieldSchema(name='uuid', dtype=milvus.DataType.VARCHAR, description='unique identifier', max_length=36,
                    is_primary=True, auto_id=True),
        milvus.FieldSchema(name=default_vec_field_name, dtype=milvus.DataType.BINARY_VECTOR, dim=default_dim),
        milvus.FieldSchema(name="imgname",dtype=milvus.DataType.VARCHAR,description = 'file path', max_length =500)
    ]
default_schema = milvus.CollectionSchema(fields=embedding_fields, description="artifact perf eval")

#Specify and load the collection in memory. 
collection = milvus.Collection(name=collection_name, schema=default_schema)
collection.load()

#Print the number of entities to make sure that collection is loaded properly. Re-check the host and port if error happens in the following line.
print("Num entities",collection.num_entities)

#Given an image path , load the image and compute PDQHash embedding for this image
def pdqhash_func(image_path):
    try:
        # Load image using PIL
        # img = Image.open(image_path).convert('RGB')
        #img = Image.open(io.BytesIO(image_path)).convert('RGB')
        image = cv2.imread(image_path)
        # Convert image to numpy array
        #img_array = np.array(img)
        # Compute PDQHash
        hash_vector, _ = pdqhash.compute(image) # Placeholder for actual PDQHash computation
        if len(hash_vector) != 256:
            print("Error: Hash vector does not have length of 256")
            return None

        # Convert binary vector to bytes using np.packbits
        bytes_data = bytes(np.packbits(hash_vector))

        return bytes_data

    except Exception as e:
        print(f"Error computing PDQHash for image '{image_path}': {e}")
        return b''


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


#Convert Hash string to binary hash 
def hex_to_hash_pdq(hexstr, hash_size=16):
    l = []
    count = hash_size * (hash_size // 4)
    if len(hexstr) != count:
        emsg = 'Expected hex string size of {}.'
        raise ValueError(emsg.format(count))
    for i in range(count // 2):
        h = hexstr[i*2:i*2+2]
        v = int("0x" + h, 16)
        l.append([v & 2**i > 0 for i in range(8)])
    val=np.array(l).flatten()#.astype(int)
    return bytes(np.packbits(val, axis=-1).tolist())


#Load the images used to query GTViz. Load the images used for evaluation and compute the query embeddings.
ff=open("query_pdqhash.txt")
data_query=[]
idx=0
import os
for f in ff:
    f=f.strip()
    path="ManipulationSource/"+f
    finalpath=""
    if os.path.exists(path):
        finalpath=path
    elif os.path.exists(path+".jpg"):
        finalpath=path+".jpg"
    elif os.path.exists(path+".png"):
        finalpath=path+".png"
    else:
        print("Image Source not found",path)
    data_query.append({"imgname":f,"pdqhash":pdqhash_func(finalpath)})
    idx+=1

#Load the images that are used to index GTVz. This is used to check if the images retrieved as matches are visual variants of query images 
#e.g. all Visual matches of img0 will have format  img0_*, img1_* similarty
#Indexed images that are not visual variants of any source image will have the format noise_*
ff=open("index_pdqhash.txt")
from collections import defaultdict
query_results=defaultdict(list)
for f in ff:
    f=f.strip()
    source=f.split("_")[0]
    query_results[source].append(f)

#Standard function to calculate F1 score given Precision and Recall 
def f1_score(precision, recall):
    """
    Calculates the F1 score given precision and recall values.

    Args:
        precision (float): The precision value.
        recall (float): The recall value.

    Returns:
        float: The F1 score.
    """
    if precision == 0 and recall == 0:
        return 0.0

    return 2 * ((precision * recall) / (precision + recall))

#Jaccard similarity function to calculate similarity between two strings 
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

#When querying Milvus, specify the type of index, and distance type used to measure similarity between two vectors 
INDEX_TYPE = 'BIN_FLAT'
METRIC_TYPE = 'HAMMING'

#Load json file updated_manipulation_ocr_en.json containing OCR output for index  (target) images, and final_source_ocr_en.json containing OCR output for source (query) images.
import json
with open('final_manipulation_ocr_en_v2.json') as f:
    manipulation_image_ocr_en = json.load(f)

with open('final_source_ocr_en.json') as f:
    source_image_ocr_en = json.load(f)

kgram=4
contextual_similarity_threshold=0.05
if 1:
    fp=0
    fn=0
    hit=0
    totaltrue=0
    precisions=[]
    recalls=[]
    totaltrues=[]
    for search in data_query:#Iterate through each of the query images from GTViz 
        query_vector=search["pdqhash"]#Get the binary vector that will be queried to Milvus  
        query_meta=search["imgname"]#Get the image path to further retrieve OCR value 
        sourcelabel=source_image_ocr_en[query_meta].replace("\n"," ").lower()#get pre-computed OCR text 
        search_params = {"metric_type": METRIC_TYPE, "params": {"radius": 91.0}}#Milvus radius based queries need  thresh+1 for similarity matching. Thus, 90+1=91

        #Query Milvus with this image vector, get as many results as possible within Radius specified 
        results = collection.search(data=[query_vector],anns_field=default_vec_field_name,
                                    param=search_params,
                                    limit=collection.num_entities,
                output_fields=["imgname"])
        #Gt contains list of matches annotated while creating GTViz. 
        gt=list(set(query_results[query_meta]))
        matched_results={}
        if 1:
            for item in results[0]:
                matched_results[item.entity.get('imgname')]=1
        matched_results=list(set(matched_results))
        #At this point, this contains only visual matches retruned by Milvus. Now we will check for contextual similarity Using OCR labels 
        img_matches[query_meta]=matched_results
        totaltrue+=len(gt)
        local_fp=0
        local_fn=0
        local_hit=0
        filtered_match_results=[]
        for rslt in matched_results:
            mylabel=manipulation_image_ocr_en[rslt].replace("\n"," ").lower()#Get the OCR label from retrieved match image
            if kgram_similarity(mylabel,sourcelabel,kgram)>=contextual_similarity_threshold:#Check for similarity between query image and retrived match image . Above similarity threshold means its a further match.
                filtered_match_results.append(rslt)
        #calculate False positive and False Negative based on ground truth and contextually filtered results
        for item in gt:
            if item not in filtered_match_results:
                fn+=1
                local_fn+=1
        for item in filtered_match_results:
            if item not in gt:
                fp+=1
                local_fp+=1
            else:
                hit+=1
                local_hit+=1
        #Compute precision recall and append to global array 
        pr=0
        rec=0
        if local_hit+local_fp>0:
            precisions.append(local_hit/(local_hit+local_fp))
            pr=local_hit/(local_hit+local_fp)
        else:
            precisions.append(0)
            pr=0
        rec=local_hit/len(gt)
        recalls.append(local_hit/len(gt))
    #Print final statistics from computed Precision Recall Arrays     
    precision=hit/(hit+fp)
    recall=hit/totaltrue
    print("Precision ",precision)
    print("Recall ",recall)
    print("F1 ",f1_score(precision,recall))
