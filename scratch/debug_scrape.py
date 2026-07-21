from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from bs4 import BeautifulSoup
import time
import re
import requests

options = Options()
options.add_argument("--headless")
service = Service("/snap/bin/geckodriver")
driver = webdriver.Firefox(options=options, service=service)

url = "https://www.flipkart.com/search?q=red+south+indian+saree"
print(f"Loading {url}...")
try:
    driver.get(url)
    time.sleep(5)
    
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    anchors = [a for a in soup.find_all("a", href=True) if "/p/" in a['href']]
    print(f"Found {len(anchors)} potential links.")
    
    seen_urls = set()
    count = 0
    for a in anchors[:15]:
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
            
        print(f"\nAnchor: {product_url}")
        if not card:
            print("  -> Skip: No card container found.")
            continue
            
        print(f"  Card tag: {card.name}, classes: {card.get('class', [])}, text len: {len(card.text)}")
        print(f"  Card HTML:\n{str(card)[:200]}")
        
        # 1. Title (Extract from URL path slug)
        title = ""
        url_path = href.split("?")[0].strip("/")
        if "/p/" in href:
            slug = url_path.split("/p/")[0].split("/")[-1]
            title = " ".join([word.capitalize() for word in slug.split("-")])
            print(f"  Extracted Title from URL slug: '{title}'")
        
        if not title:
            img = card.find("img")
            if img and img.get("alt"):
                title = img.get("alt").strip()
            if not title and a.text.strip():
                title = a.text.strip()
            
        print(f"  Title: '{title}'")
        if not title or len(title) < 15:
            print("  -> Skip: Title missing or too short.")
            continue
            
        # 2. Image URL
        image_url = ""
        imgs = card.find_all("img")
        print(f"  Found {len(imgs)} images in card.")
        for im in imgs:
            src = im.get("src", "")
            print(f"    Img src: '{src}'")
            if src.startswith("http") and "rukminim" in src:
                image_url = src
                break
                
        print(f"  Selected Image URL: '{image_url}'")
        if not image_url:
            print("  -> Skip: No valid image URL starting with http and containing 'rukminim'.")
            continue
            
        prices_raw = re.findall(r"₹(\d{1,3}(?:,\d{3})*)", card.text)
        print(f"  Prices found: {prices_raw}")
        if not prices_raw:
            print("  -> Skip: No prices found.")
            continue
            
        count += 1
        print("  >> SUCCESS: Valid product parsed!")
        if count >= 3:
            break
            
except Exception as e:
    print("Selenium Error:", str(e))
finally:
    driver.quit()
