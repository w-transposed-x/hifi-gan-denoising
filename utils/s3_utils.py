#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:05:51 2021

@author: Wilson Ye
"""
import io
import os
import boto3
import botocore

"""
Note: everything inside a s3 butcket is an object, including a folder, identified
by a key/path.
"""


#check if path of an object exists in s3 bucket
def is_path_existed(key_path):
    s3_client = boto3.client(
                        's3',
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), 
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), 
                        region_name=os.getenv('AWS_DEFAULT_REGION'))
     
    # These define the bucket and object to read
    bucket = 'speakupai-s3-bucket'
    
    
    objects = None
    
    try:
        objects = s3_client.list_objects_v2(Bucket=bucket, Prefix=key_path)["Contents"]
    except botocore.exceptions.ClientError:
        return False
    else:
        if objects:
            return True
        else:
            return False

#get paths for all files inside an object/folder
def load_file_paths(key_path): 
        
    s3_client = boto3.client(
                        's3',
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), 
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), 
                        region_name=os.getenv('AWS_DEFAULT_REGION'))
     
    # These define the bucket and object to read
    bucket = 'speakupai-s3-bucket'
    
   
    
    
    #list objects
    objects = s3_client.list_objects_v2(Bucket=bucket, Prefix=key_path)["Contents"]

    file_list = [obj['Key'] for obj in objects]

    return file_list


#load audio object by path via soundfile
def load_audio_object(key_path):
    
    s3_client = boto3.client(
                        's3',
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), 
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), 
                        region_name=os.getenv('AWS_DEFAULT_REGION'))
     
    # These define the bucket and object to read
    bucket = 'speakupai-s3-bucket'
        
    #Create a file object using the bucket and object key. 
    file_object = s3_client.get_object(Bucket=bucket, Key=key_path) 
    
    # # open the file object and read it into the variable filedata.
    # file data will be a binary stream.  If you want to decode it, you can add .decode('utf-8') after read
    file = file_object['Body'].read()
    
    return io.BytesIO(file)
