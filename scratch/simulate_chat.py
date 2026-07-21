import requests
import json

url = "http://localhost:8000/api/chat"
payload = {
    "query": "i am planning mahabaleshwar trip with my ffriends",
    "history": []
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    data = response.json()
    print("\nResponse Text:\n", data.get("response"))
    print("\nProducts:")
    for idx, p in enumerate(data.get("products", [])):
        print(f"#{idx+1}: {p.get('product_title')} | Brand: {p.get('brand')} | Subcat: {p.get('subcategory')} | Price: {p.get('selling_price')} | Score: {p.get('final_ranking_score')}")
except Exception as e:
    print("Error calling endpoint:", str(e))
