import requests
from fastapi import HTTPException

# https://www.authenticatorapi.com/Validate.aspx?Pin=379567&SecretCode=lovable&AppName=theia-api&AppInfo=Thomi
def authenticate(pin: str):
    res = requests.get(
        "https://www.authenticatorapi.com/Validate.aspx",
        params={
            "Pin": pin,
            "SecretCode": "lovable",
            "AppName": "theia-api",
            "AppInfo": "Thomi"
        }
    )

    # Check if res is True
    if res.text != "True":
        raise HTTPException(status_code=401, detail="Invalid pin")