# Pixelmod
Source code and labeled dataset for our Usenix Security'24 paper PixelMoD

# Instructions to Download and Run PixelMod 
## Step 1 
Download the evaluation docker image , and volumes associated with Docker image from the  [Zenodo URL.](https://zenodo.org/records/12570381)

(Note: due to refactoring for evaluation workflow, the stable Zenodo URL is updated from the one initially submitted during the AE process. )

List of tarballs to download include: 

 1. `pixelmod.tar.gz` ( Docker Image containing Streamlit and benchmarking script )
 2. `etcd_backup.tar.gz` (Docker volume for service **etcd**) 
 3. `milvus.tar.gz` (Docker volume for service **Milvus**)
 4. `minio.tar.gz`( Docker volume for service **Minio**)
 
 Download these tar files and place them in the same subdirectory.

---
# Step 2 
### 2.1 Importing Docker Image 
First, run the following command to load the docker image containing PixelMod Evaluation infrastructure

    sudo docker load < pixelmod.tar.gz

This will load the evaluation image. You can run the following command and expect to see `milvus-pixelmod:latest` in the list of docker images listed.

    sudo docker image ls

### 2.2 Importing Docker Volumes 
In the same directory the three zipped files containing the volumes are located, run the  following three commands to load the docker volumes necessary.

 1. `sudo bash import_volume.sh milvus.tar.gz milvus`
 2. `sudo bash import_volume.sh etcd.tar.gz etcd`
 3. `sudo bash import_volume.sh minio.tar.gz minio`

Run the following command to make sure the volumes are loaded properly 

    sudo docker volume ls

### 2.3 Docker compose  
Finally, run the command `docker compose up -d` to start all the containers. Make sure to download the file `docker-compose.yml` on the companion repository.
This should start four different containers: `milvus-standalone`, `milvus-etcd`, `milvus-minio` and `milvus-pixelmod`. 


## Step3.  Artifact Evaluation 

### 3.1 Basic Test
Make sure the container and Milvus index is working properly by running the command `sudo docker exec -it  $(sudo docker container ls  | grep 'milvus-pixelmod' | awk '{print $1}') python3 basic_test.py`

#### 3.2 Interacting with companion streamlit web application 
Starting the container automatically runs the streamlit web application in port 8501 of the container. Port 8501 of the container is mapped to port 8501 of host machine (evaluators can modify this on `client` service of the compose file to change the mapping). 
Therefore, browsing to  `localhost:8501`  on the host machine should start the streamlit application succesfully.

#### 3.3 Running experiment2_evaluating.py for benchmarking PixelMod.
Users can run the command `sudo docker exec -it  $(sudo docker container ls  | grep 'milvus-pixelmod' | awk '{print $1}') /bin/bash` to start a bash terminal in the running container where `milvus-pixelmod:latest` is the container imported in Section 2.1. 
Evaluation can then be performed by running the script `python experiment2_evaluating.py`
 ---

### Citing The Paper
```
@inproceedings {sec24:pixelmod,
    title = {PIXELMOD: Improving Soft Moderation of Visual Misleading Information on Twitter},
    booktitle = {33rd USENIX Security Symposium (USENIX Security)},
    year = {2024},
    author={Pujan Paudel, Chen Ling, Jeremy Blackburn, and Gianluca Stringhini}
}
