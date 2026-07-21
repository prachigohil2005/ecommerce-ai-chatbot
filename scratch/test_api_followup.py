import requests
import json

url = "http://localhost:8000/api/chat"

def test_api():
    # Turn 1
    p1 = {
        "query": "i am planing manali trip for traking",
        "history": []
    }
    print("Sending query 1: 'i am planing manali trip for traking'...")
    res1 = requests.post(url, json=p1).json()
    print("\nTurn 1 Response:")
    print(res1.get("response"))
    print("\nTurn 1 Products:")
    for p in res1.get("products", []):
        print(f"- {p['product_title']} | Category: {p['category']} | Price: {p['selling_price']}")
        
    # Turn 2
    history = [
        {"role": "user", "content": "i am planing manali trip for traking"},
        {"role": "assistant", "content": res1.get("response")}
    ]
    p2 = {
        "query": "suggest clothes also",
        "history": history
    }
    print("\nSending query 2: 'suggest clothes also'...")
    res2 = requests.post(url, json=p2).json()
    print("\nTurn 2 Response:")
    print(res2.get("response"))
    print("\nTurn 2 Products:")
    for p in res2.get("products", []):
         print(f"- {p['product_title']} | Category: {p['category']} | Price: {p['selling_price']}")

if __name__ == "__main__":
    test_api()
