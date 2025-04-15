import boto3
import os

# Configure LocalStack
os.environ['AWS_ACCESS_KEY_ID'] = 'test'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'test'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
localstack_endpoint = 'http://localhost:4566'

# Create S3 client
s3 = boto3.client('s3', endpoint_url=localstack_endpoint)

# Create a bucket
bucket_name = 'your-bucket'

# List files in the bucket
response = s3.list_objects_v2(Bucket=bucket_name)

if 'Contents' in response:
    for obj in response['Contents']:
        print(obj['Key'])
else:
    print("Bucket is empty or does not exist.")