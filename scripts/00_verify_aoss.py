#Code to check connectivity with the AWS resources 
import os
import boto3
import requests
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv

#Load the .env environment variables.
load_dotenv()

endpoint=os.getenv("OPENSEARCH_END_POINT")
if not endpoint:
    raise SystemExit("Configure OPENSEARCH_END_POINT")


region = os.getenv("AWS_REGION", "us-east-1")
service = "aoss"  # IMPORTANT: OpenSearch Serverless uses 'aoss' for SigV4


session = boto3.Session()
creds = session.get_credentials().get_frozen_credentials()
auth = AWS4Auth(creds.access_key, creds.secret_key, region, service, session_token=creds.token)

index = "rag-docs"
url = f"{endpoint}/{index}/_count"

resp = requests.get(url, auth=auth, timeout=30)
print("URL:", url)
print("Status:", resp.status_code)
print("Body:", resp.text)