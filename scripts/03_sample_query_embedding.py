import boto3, json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

query = "What does the book say about anxiety and focusing on the present?"

body = {
    "inputText": query,
    "dimensions": 512,
    "normalize": True
}

resp = client.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps(body),
)

embedding = json.loads(resp["body"].read())["embedding"]

print(len(embedding))
print(embedding)   # sanity check