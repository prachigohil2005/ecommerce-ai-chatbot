import requests
from bs4 import BeautifulSoup
import re
import random
import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

def debug_scrape():
    options = Options()
    options.add_argument("--headless")
    service = Service(executable_path="/snap/bin/geckodriver")
    
    driver = webdriver.Firefox(options=options, service=service)
    url = "https://www.flipkart.com/search?q=pink%20lehenga%2037500"
    driver.get(url)
    time.sleep(3)
    
    soup = BeautifulSoup(driver.page_source, "html.parser")
    anchors = [a for a in soup.find_all("a", href=True) if "/p/" in a['href']]
    
    print(f"Found {len(anchors)} product links.")
    
    seen_urls = set()
    for idx, a in enumerate(anchors[:3]):
        href = a['href']
        product_url = "https://www.flipkart.com" + href.split("?")[0]
        if product_url in seen_urls:
            continue
        seen_urls.add(product_url)
        
        parent = a.parent
        card = None
        while parent and parent.name != 'body':
            if "₹" in parent.text and len(parent.text) < 1500:
                card = parent
            parent = parent.parent
            
        if card:
            print(f"\n--- CARD {idx+1} TEXT ---")
            print(card.text.strip())
            prices_raw = re.findall(r"₹(\d{1,3}(?:,\d{3})*)", card.text)
            print("Extracted prices_raw:", prices_raw)
            
    driver.quit()

if __name__ == "__main__":
    debug_scrape()
