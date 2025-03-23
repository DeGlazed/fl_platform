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

# Create a dummy file
file_name = 'dummy_file.txt'
with open(file_name, 'w') as f:
    f.write('This is a dummy file.')

# Upload the dummy file to the bucket
s3.upload_file(file_name, bucket_name, file_name)

# Download the file back from the bucket
downloaded_file_name = 'downloaded_' + file_name
s3.download_file(bucket_name, file_name, downloaded_file_name)

# Verify the file content
with open(downloaded_file_name, 'r') as f:
    content = f.read()
    print(content)