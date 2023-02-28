import json
import base64


def extract_email_from_globus_jwt(jwt: str) -> str:
    try:
        _, payload_b64, _ = jwt.split(".")
        payload = json.loads(base64.b64decode(payload_b64 + "===").decode("utf-8"))
    except Exception as e:
        raise Exception('Invalid JWT') from e
    try:
        email = payload['identity_set'][0]['email']
    except KeyError as e:
        raise Exception('JWT did not include user email') from e
    return email
