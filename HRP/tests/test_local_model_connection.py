import requests

response = requests.post(
    "http://127.0.0.1:11434/api/generate",
    json={
        "model": "qwen3:8b",
        "prompt": (
            "Explain in simple terms why elevated "
            "creatinine might influence a hospital "
            "readmission risk prediction."
        ),
        "stream": False,
    },
    timeout=180,
)

print(response.json()["response"])