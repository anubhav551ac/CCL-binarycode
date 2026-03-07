import requests

def send_pushover_notification(user_key, api_token, message):
    url = "https://api.pushover.net/1/messages.json"
    data = {
        "token": api_token,
        "user": user_key,
        "message": message,
        "title": "GhostGrid Sentinel",
        "priority": 0,   
        "sound": "siren" 
    }
    
    response = requests.post(url, data=data)

USER_KEY = "ug4i6e1oki9o4wrr4iq1nd7hhuxeye"
API_TOKEN = "ai62ez85469ho238mv1rf5mgg662pn"

if __name__ == "__main__":
    test_msg = f"Device is unattended"
    send_pushover_notification(USER_KEY, API_TOKEN, test_msg)