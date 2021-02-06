#!/usr/bin/env python3
import boto3
import botocore
import os
import pickle
import json
import h5py
from io import BytesIO

from config import *

######################################### S3 #########################################
def upload_to_s3(bucket_name, folder, file_name):
    s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    bucket = s3.Bucket(bucket_name)
    
    print(f'Uploading {file_name} to bucket {bucket_name}')
    with open(os.path.join(folder, file_name), 'rb') as data: 
        response = bucket.Object(file_name).put(Body=data)

    print(f"Operation result {response['ResponseMetadata']['HTTPStatusCode']}")

          
def download_pickle_from_s3(bucket_name, folder, file_name):
    s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    bucket = s3.Bucket(bucket_name)    

    print(f'Downloading {file_name} from bucket {bucket_name}')
    response = bucket.Object(file_name).get()
    data = pickle.load(BytesIO(response['Body'].read()))
    
    with open(os.path.join(folder, file_name), 'wb') as f:
        pickle.dump(data, f)    
    
    return data

def download_json_from_s3(bucket_name, folder, file_name):
    s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    bucket = s3.Bucket(bucket_name)    

    print(f'Downloading {file_name} from bucket {bucket_name}')
    response = bucket.Object(file_name).get()
    json_content = json.loads(response['Body'].read().decode('utf-8'))
    
    with open(os.path.join(folder, file_name), 'w') as outfile:
        json.dump(json_content, outfile)  

    return json_content

def download_h5py_from_s3(bucket_name, folder, file_name):
    s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    bucket = s3.Bucket(bucket_name)    

    print(f'Downloading {file_name} from bucket {bucket_name}')
    response = bucket.Object(file_name).get()
    data = response['Body'].read()
    
    with open(os.path.join(folder, file_name), 'wb') as outfile:
        outfile.write(data)  

    return data    