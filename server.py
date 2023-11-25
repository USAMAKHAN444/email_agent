import requests

print(

requests.post(

"http://127.0.0.1:8000",

json={ "query": "I want to gives and bussiness chatbot ideas?"

}

).json()

)