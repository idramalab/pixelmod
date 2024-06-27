import pymilvus as milvus
import cv2
from pymilvus import Collection
hostname="standalone"#127.0.0.1 for testing locally
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



INDEX_TYPE = 'BIN_FLAT'
METRIC_TYPE = 'HAMMING'

print("Testing Index with 5 sampled random images")
import random
sampled=random.sample(data_query,5)

radius=32
if 1:
    for search in sampled:
        query_vector=search["pdqhash"]
        query_meta=search["imgname"]
        print("Query Image ",query_meta, " searched with Hamming Distance of ",radius)
        search_params = {"metric_type": METRIC_TYPE, "params": {"radius": radius+1}}#Seems like you need thresh+1
        results = collection.search(data=[query_vector],anns_field=default_vec_field_name,
                                    param=search_params,
                                    limit=collection.num_entities,
                output_fields=["imgname"])
        matched_results=[]
        if 1:
            for rslt in results[0]:
                matched_results.append({"image":rslt.entity.get('imgname'),"distance":
                    rslt.distance})
        print("Returned ",len(matched_results)," images ")
        print(matched_results) 
        print("-------")
