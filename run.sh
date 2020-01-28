#! /bin/bash
# Script for model training and result transfer to s3
TOP_DIRECTORY="omip_dl"
BUCKET_NAME="omip-images"
# Activate conda env
source activate pytorch_p36
python manage.py
# copy results to s3
aws s3 cp --recursive ~/$TOP_DIRECTORY/models/ s3://$BUCKET_NAME/models/