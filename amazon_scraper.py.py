import os
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load headers and cookies from .env
load_dotenv()
HEADERS = json.loads(os.getenv("HEADERS"))
COOKIES = json.loads(os.getenv("COOKIES"))

# Get the search keyword dynamically
search_keyword = input("Enter the product keyword to scrape (e.g., watches, mobiles, shoes): ").strip()
BASE_URL = f"https://www.amazon.in/s?k={search_keyword}"

def fetch_content(url):
    try:
        response = requests.get(url, headers=HEADERS, cookies=COOKIES, timeout=10)
        response.raise_for_status()
        print(f"✅ Successfully fetched: {url}")
        return BeautifulSoup(response.content, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching page: {e}")
        return None

def get_title(item):
    title = item.find("h2", class_="a-size-medium a-spacing-none a-color-base a-text-normal")
    return title.text.strip() if title else None

def get_brand(title_text):
    if not title_text:
        return None
    if title_text.lower().startswith("soundcore"):
        return "Anker"
    return title_text.split()[0]

def get_price(item):
    discount_price = item.find("span", class_="a-price")
    return (
        discount_price.find("span", class_="a-offscreen").text.strip()
        if discount_price and discount_price.find("span", class_="a-offscreen")
        else None
    )

def get_mrp(item):
    base_price = item.find("div", class_="a-section aok-inline-block")
    return (
        base_price.find("span", class_="a-offscreen").text.strip()
        if base_price and base_price.find("span", class_="a-offscreen")
        else None
    )

def get_discount_percentage(item):
    discount = item.find("span", string=lambda text: text and "%" in text)
    return discount.text.strip().strip("()") if discount else None

def get_rating(item):
    rating = item.find("span", class_="a-icon-alt")
    return rating.text.strip() if rating else None

def get_reviews(item):
    reviews = item.find("span", class_="a-size-base s-underline-text")
    return reviews.text.strip() if reviews else None

def get_product_id(item):
    return item.get("data-asin", None)

def get_product_link(item):
    link = item.find("a", class_="a-link-normal s-no-outline")
    return "https://www.amazon.in" + link["href"] if link and "href" in link.attrs else None

def parse_product(item, category):
    try:
        title_text = get_title(item)
        return {
            "Product": title_text,
            "Category": category,
            "Price": get_price(item),
            "Discount": get_discount_percentage(item),
            "Promotion": "No promotion",
            "Rating": get_rating(item),
            "Reviews": get_reviews(item),
            "Source": "Amazon"
        }
    except Exception as e:
        print(f"⚠️ Error parsing product: {e}")
        return None

def scrape_page(url, category):
    soup = fetch_content(url)
    if not soup:
        return [], None
    items = soup.find_all("div", {"data-component-type": "s-search-result"})
    products = [parse_product(item, category) for item in items if parse_product(item, category)]
    next_button = soup.find("a", class_="s-pagination-next")
    next_page_url = "https://www.amazon.in" + next_button["href"] if next_button and "href" in next_button.attrs else None
    return products, next_page_url

def scrape_within_time(base_url, category, max_time_minutes=5):
    all_products = []
    current_page = 1
    next_page_url = base_url
    end_time = datetime.now() + timedelta(minutes=max_time_minutes)

    print(f"🚀 Scraping for {max_time_minutes} minutes...")
    while next_page_url and datetime.now() < end_time:
        print(f"\n⏳ Scraping page {current_page}...")
        products, next_page_url = scrape_page(next_page_url, category)
        if not products:
            print("⚠️ No more products found.")
            break
        all_products.extend(products)
        print(f"✅ Scraped {len(products)} products from page {current_page}.")
        current_page += 1
        time.sleep(2)
    print("🏁 Scraping finished.")
    return all_products

def save_to_csv(data, filename="amazon_prices.csv"):
    try:
        os.makedirs("Amazon", exist_ok=True)
        full_file_path = os.path.join("Amazon", filename)
        df = pd.DataFrame(data)
        df.to_csv(full_file_path, index=False)
        print(f"💾 Data saved to {full_file_path} with {len(data)} rows.")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")

if __name__ == "__main__":
    max_scraping_time = 1  # minutes
    products = scrape_within_time(BASE_URL, search_keyword, max_time_minutes=max_scraping_time)
    if products:
        save_to_csv(products, filename=f"amazon_{search_keyword}_prices.csv")
        print(f"Scraped {len(products)} products for keyword '{search_keyword}'.")
    else:
        print("No products found.")
