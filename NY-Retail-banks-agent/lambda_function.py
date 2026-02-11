import json
import boto3

client = boto3.client("dynamodb")

def lambda_handler(event, context):
    print(f"This is the input from agent{event}")

    # 1) Direct invocation payloads (Lambda console tests)
    account_id = event.get("account_id")
    if account_id is None:
        account_id = event.get("AccountID")

    # 2) Bedrock Agents action-group payloads
    if account_id is None:
        params = event.get("parameters") or {}

        if isinstance(params, list):
            account_id = next(
                (p.get("value") for p in params if p.get("name") in ("account_id", "AccountID")),
                None
            )
        elif isinstance(params, dict):
            v = params.get("account_id") or params.get("AccountID")
            account_id = v.get("value") if isinstance(v, dict) else v

    if account_id is None or str(account_id).strip() == "":
        raise ValueError(f"AccountID is missing. Event keys: {list(event.keys())}")

    # DynamoDB Number ("N") MUST be a string
    account_id_str = str(account_id)

    response = client.get_item(
        TableName="CustomerAccountStatus",
        Key={"AccountID": {"N": account_id_str}}
    )

    response_body = {
        "application/json": {
            "body": json.dumps(response)
        }
    }

    action_response = {
        "actionGroup": event.get("actionGroup", "CustomerAccountStatus"),
        "apiPath": event.get("apiPath", "/getAccountStatus"),
        "httpMethod": event.get("httpMethod", "POST"),
        "httpStatusCode": 200,
        "responseBody": response_body
    }

    return {
        "messageVersion": "1.0",
        "response": action_response,
        "sessionAttributes": event.get("sessionAttributes", {}),
        "promptSessionAttributes": event.get("promptSessionAttributes", {})
    }
