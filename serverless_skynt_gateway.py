import json

def lambda_handler(event, context):
    # AWS Lambda implementation
    # Your code for processing events here
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from AWS!')
    }


def gcp_handler(request):
    # GCP Cloud Functions implementation
    # Your code for processing requests here
    return 'Hello from GCP!'


def azure_handler(req):
    # Azure Functions implementation
    # Your code for processing requests here
    return 'Hello from Azure!'
