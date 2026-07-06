from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from bs4 import BeautifulSoup
import time
import re

options = Options()
options.add_argument("--headless")
service = Service("/snap/bin/geckodriver")
driver = webdriver.Firefox(options=options, service=service)

url = "https://www.flipkart.com/search?q=smartphones"
print(f"Loading {url}...")
try:
    driver.get(url)
    time.sleep(5)
    
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    # Find all divs containing products. Typically, search items have a distinct structure.
    # Let's search for standard product container elements.
    # On row layout: it's a div containing product anchors
    product_containers = []
    
    # Let's find anchors with "/p/" and trace up to find their common container parent
    anchors = [a for a in soup.find_all("a", href=True) if "/p/" in a['href'] and a.text.strip()]
    if anchors:
        print(f"Inspecting container structure for {anchors[0]['href']}")
        # Let's get the parent of this anchor
        parent = anchors[0].parent
        while parent and parent.name != 'body':
            # Check if this parent contains text containing "₹"
            p_text = parent.text
            if "₹" in p_text and len(p_text) < 1500:
                print(f"\nParent container tag: {parent.name}, classes: {parent.get('class', [])}")
                print(f"Parent content text:\n{parent.text.strip()[:600]}")
                
                # Check for images inside this container
                imgs = parent.find_all("img")
                for i, img in enumerate(imgs):
                    print(f"  Img {i+1} src: {img.get('src')}")
                    print(f"  Img {i+1} alt: {img.get('alt')}")
                break
            parent = parent.parent
            
except Exception as e:
    print("Selenium Error:", str(e))
finally:
    driver.quit()
