"""
scraper_agent.py — ScraperAgent
Responsible for:
- web scraping using requests + BeautifulSoup
- optional Playwright support
- mock market data generation
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
from datetime import datetime


class ScraperAgent:

    def __init__(self, output_dir="outputs"):

        self.name = "ScraperAgent"
        self.output_dir = output_dir
        self.scraped_data = []

        os.makedirs(output_dir, exist_ok=True)

        print(f"[{self.name}] Agent initialized")

    # ---------------------------------------------------------
    # REQUESTS + BEAUTIFULSOUP SCRAPER
    # ---------------------------------------------------------
    def scrape_with_requests(self, url, tag="p", limit=10):

        print(f"[{self.name}] Scraping using requests: {url}")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0"
            )
        }

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=15
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            elements = soup.find_all(tag)[:limit]

            texts = [
                el.get_text(strip=True)
                for el in elements
                if el.get_text(strip=True)
            ]

            print(f"[{self.name}] Scraped {len(texts)} elements")

            return texts

        except Exception as e:

            print(f"[{self.name}] Scraping failed: {e}")

            return []

    # ---------------------------------------------------------
    # PLAYWRIGHT SCRAPER
    # ---------------------------------------------------------
    async def scrape_with_playwright(self, url):

        try:

            from playwright.async_api import async_playwright

            print(f"[{self.name}] Scraping using Playwright: {url}")

            async with async_playwright() as p:

                browser = await p.chromium.launch(
                    headless=True
                )

                page = await browser.new_page()

                await page.goto(
                    url,
                    wait_until="networkidle",
                    timeout=30000
                )

                content = await page.content()

                await browser.close()

                soup = BeautifulSoup(
                    content,
                    "html.parser"
                )

                text = soup.get_text(
                    separator=" ",
                    strip=True
                )[:5000]

                print(f"[{self.name}] Playwright scraped text")

                return text

        except ImportError:

            print(f"[{self.name}] Playwright not installed")

            return ""

        except Exception as e:

            print(f"[{self.name}] Playwright error: {e}")

            return ""

    # ---------------------------------------------------------
    # MOCK MARKET DATA
    # ---------------------------------------------------------
    def generate_mock_scraped_data(self):

        print(f"[{self.name}] Generating mock scraped market data...")

        mock_products = [

            {
                "product": "Laptop Pro X",
                "price": 1299.99,
                "rating": 4.5,
                "reviews": 2341,
                "category": "Electronics"
            },

            {
                "product": "Wireless Earbuds",
                "price": 89.99,
                "rating": 4.3,
                "reviews": 5672,
                "category": "Electronics"
            },

            {
                "product": "Smart Watch Ultra",
                "price": 399.99,
                "rating": 4.7,
                "reviews": 1893,
                "category": "Electronics"
            },

            {
                "product": "Running Shoes X9",
                "price": 149.99,
                "rating": 4.6,
                "reviews": 3211,
                "category": "Sports"
            },

            {
                "product": "Coffee Maker Pro",
                "price": 74.99,
                "rating": 4.2,
                "reviews": 1102,
                "category": "Home & Garden"
            },

            {
                "product": "Python Cookbook",
                "price": 45.00,
                "rating": 4.8,
                "reviews": 892,
                "category": "Books"
            },

            {
                "product": "Yoga Mat Premium",
                "price": 59.99,
                "rating": 4.4,
                "reviews": 2108,
                "category": "Sports"
            },

            {
                "product": "LED Desk Lamp",
                "price": 39.99,
                "rating": 4.1,
                "reviews": 764,
                "category": "Home & Garden"
            },

            {
                "product": "Mechanical Keyboard",
                "price": 189.99,
                "rating": 4.6,
                "reviews": 3456,
                "category": "Electronics"
            },

            {
                "product": "Water Bottle Insulated",
                "price": 29.99,
                "rating": 4.5,
                "reviews": 7823,
                "category": "Sports"
            }
        ]

        df = pd.DataFrame(mock_products)

        df["scraped_at"] = datetime.now().isoformat()

        df["source"] = "mock_market_data"

        path = os.path.join(
            self.output_dir,
            "scraped_market_data.json"
        )

        with open(path, "w") as f:
            json.dump(mock_products, f, indent=2)

        print(f"[{self.name}] Mock data saved -> {path}")

        return df

    # ---------------------------------------------------------
    # AGENT MESSAGE
    # ---------------------------------------------------------
    def get_agent_message(self, df):

        return {
            "from_agent": self.name,
            "status": "complete",
            "scraped_df": df,
            "row_count": len(df),
        }