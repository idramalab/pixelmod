import pymilvus as milvus
import cv2
from pymilvus import Collection
hostname="127.0.0.1"#127.0.0.1 for testing locally
client=milvus.connections.connect(host=hostname, port=19530)

collection_name="usenixtest_pdqhash_v2"
default_dim=256
default_vec_field_name="pdqhash"

embedding_fields = [
        milvus.FieldSchema(name='uuid', dtype=milvus.DataType.VARCHAR, description='unique identifier', max_length=36,
                    is_primary=True, auto_id=True),
        milvus.FieldSchema(name=default_vec_field_name, dtype=milvus.DataType.BINARY_VECTOR, dim=default_dim),
        milvus.FieldSchema(name="imgname",dtype=milvus.DataType.VARCHAR,description = 'file path', max_length =500)
    ]


default_schema = milvus.CollectionSchema(fields=embedding_fields, description="artifact perf eval")


collection = milvus.Collection(name=collection_name, schema=default_schema)

collection.load()

print("Num entities",collection.num_entities)


import pdqhash
import numpy as np
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


from PIL import Image
import io



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


ff=open("index_pdqhash.txt")
from collections import defaultdict
query_results=defaultdict(list)
for f in ff:
    f=f.strip()
    source=f.split("_")[0]
    query_results[source].append(f)

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





INDEX_TYPE = 'BIN_FLAT'
METRIC_TYPE = 'HAMMING'

#Load json file updated_manipulation_ocr_en.json,
import json
with open('final_manipulation_ocr_en_v2.json') as f:
    manipulation_image_ocr_en = json.load(f)

with open('final_source_ocr_en.json') as f:
    source_image_ocr_en = json.load(f)

error_missing_ocr=[]
empty_label=[]
nonempty_label=0
real_empty_label=0
contextual_filter=0
img_matches=defaultdict(list)
if 1:
    fp=0
    fn=0
    hit=0
    totaltrue=0
    precisions=[]
    recalls=[]
    totaltrues=[]
    for search in data_query:
        query_vector=search["pdqhash"]
        query_meta=search["imgname"]
        sourcelabel=source_image_ocr_en[query_meta].replace("\n"," ").lower()
        search_params = {"metric_type": METRIC_TYPE, "params": {"radius": 91.0}}#Seems like you need thresh+1
     
        results = collection.search(data=[query_vector],anns_field=default_vec_field_name,
                                    param=search_params,
                                    limit=collection.num_entities,
                output_fields=["imgname"])
        gt=list(set(query_results[query_meta]))
        matched_results={}
        #break
        if 1:
            for item in results[0]:
                matched_results[item.entity.get('imgname')]=1
        matched_results=list(set(matched_results))
        img_matches[query_meta]=matched_results
        totaltrue+=len(gt)
        local_fp=0
        local_fn=0
        local_hit=0
        #Create a filter of matched_results, not just matched results
        filtered_match_results=[]
        for rslt in matched_results:
            mylabel=manipulation_image_ocr_en[rslt].replace("\n"," ").lower()
            if kgram_similarity(mylabel,sourcelabel,4)>=0.05:
                filtered_match_results.append(rslt)
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
    precision=hit/(hit+fp)
    recall=hit/totaltrue
    print("Precision ",precision)
    print("Recall ",recall)
    print("F1 ",f1_score(precision,recall))
