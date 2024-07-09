#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <backup_file> <volume_name>"
    exit 1
fi

BACKUP_FILE=$1
VOLUME_NAME=$2

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file '$BACKUP_FILE' not found."
    exit 1
fi

# Check if volume exists, create it if it doesn't
if ! docker volume inspect "$VOLUME_NAME" &> /dev/null; then
    echo "Volume '$VOLUME_NAME' does not exist. Creating it..."
    docker volume create "$VOLUME_NAME"
fi

# Create a temporary container with the volume mounted
TEMP_CONTAINER=$(docker create -v ${VOLUME_NAME}:/data alpine:latest /bin/true)

# Restore the backup to the volume
docker run --rm --volumes-from ${TEMP_CONTAINER} -v $(pwd):/backup alpine:latest sh -c "cd /data && tar xzvf /backup/${BACKUP_FILE} --strip 1"

# Remove the temporary container
docker rm ${TEMP_CONTAINER}

echo "Backup restored to volume: ${VOLUME_NAME}"
