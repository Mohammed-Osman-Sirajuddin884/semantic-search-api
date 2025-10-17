import requests

url = "http://127.0.0.1:8000/similarity"
data = {
    "docs": [
        "FastAPI makes APIs easy",
        "Flask is a lightweight framework",
        "Django is full-featured"
    ],
    "query": "I like building APIs"
}
response = requests.post(url, json=data)
print(response.status_code, response.text)
