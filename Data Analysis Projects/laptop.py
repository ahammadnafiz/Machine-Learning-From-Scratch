from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from fake_useragent import UserAgent
import csv
import time
import os
import random

# List of proxy servers (you should replace these with actual proxies)
PROXIES = [
    '123.45.67.89:8080',
    '98.76.54.32:3128',
    '137.66.36.81:80',
    '1.2.210.13:8080',
    '136.226.245.22:8080',
    
    # Add more proxies here
]

def get_random_proxy():
    return random.choice(PROXIES)

def get_random_user_agent():
    ua = UserAgent()
    return ua.random

def setup_driver(use_proxy=True):
    options = webdriver.EdgeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--log-level=3')  # fatal only
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    # Add random user agent
    options.add_argument(f'user-agent={get_random_user_agent()}')
    
    # Add proxy if enabled
    if use_proxy:
        proxy = get_random_proxy()
        options.add_argument(f'--proxy-server={proxy}')
    
    service = Service('C:/edgedriver_win64/msedgedriver.exe')  # Update this path
    
    if os.name == 'posix':  # for Mac/Linux
        return webdriver.Edge(service=service, options=options, service_args=['--quiet'])
    elif os.name == 'nt':  # for Windows
        return webdriver.Edge(service=service, options=options)

def extract_product_info(driver, url, max_retries=3, timeout=20):
    for attempt in range(max_retries):
        try:
            driver.get(url)
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.ID, "productTitle")))
            
            product = {}
            product['title'] = driver.find_element(By.ID, "productTitle").text.strip()
            
            try:
                price_element = driver.find_element(By.CLASS_NAME, "a-price-whole")
                price_fraction = driver.find_element(By.CLASS_NAME, "a-price-fraction")
                product['price'] = f"{price_element.text}.{price_fraction.text}"
            except NoSuchElementException:
                product['price'] = "N/A"
            
            details_to_extract = [
                "Standing screen display size", "Screen Resolution", "Processor", "RAM", 
                "Hard Drive", "Graphics Coprocessor", "Chipset Brand", "Wireless Type", 
                "Average Battery Life (in hours)", "Brand", "Series", "Operating System", 
                "Item Weight", "Color", "Processor Brand", "Number of Processors", "Batteries"
            ]
            
            for section_id in ["productDetails_techSpec_section_1", "productDetails_techSpec_section_2"]:
                try:
                    details_section = driver.find_element(By.ID, section_id)
                    rows = details_section.find_elements(By.TAG_NAME, "tr")
                    for row in rows:
                        try:
                            key = row.find_element(By.TAG_NAME, "th").text.strip()
                            if key in details_to_extract:
                                value = row.find_element(By.TAG_NAME, "td").text.strip()
                                product[key] = value
                        except NoSuchElementException:
                            continue
                except NoSuchElementException:
                    print(f"Could not find product details section {section_id} for: {url}")
            
            return product
        
        except TimeoutException:
            print(f"Timeout on attempt {attempt + 1} for URL: {url}")
            if attempt == max_retries - 1:
                print(f"Max retries reached. Skipping URL: {url}")
                return None
            time.sleep(5 * (attempt + 1))  # Exponential backoff
        
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1} for URL: {url}")
            print(f"Error: {str(e)}")
            if attempt == max_retries - 1:
                print(f"Max retries reached. Skipping URL: {url}")
                return None
            time.sleep(5 * (attempt + 1))  # Exponential backoff

def scrape_laptops(base_url, num_pages):
    laptops = []
    page = 1

    while page <= num_pages:
        # Create a new driver for each page to rotate proxy and user agent
        driver = setup_driver()
        url = f"{base_url}&page={page}"
        print(f"Scraping page {page}")
        
        try:
            driver.get(url)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.s-main-slot")))
        except TimeoutException:
            print(f"Timeout waiting for page {page} to load")
            driver.quit()
            page += 1
            continue

        laptop_links = driver.find_elements(By.CSS_SELECTOR, "a.a-link-normal.s-underline-text.s-underline-link-text.s-link-style.a-text-normal")
        links = [link.get_attribute('href') for link in laptop_links if link.get_attribute('href')]

        for i, link in enumerate(links, 1):
            print(f"Scraping laptop {i} of {len(links)} on page {page}")
            laptop_info = extract_product_info(driver, link)
            if laptop_info:
                laptops.append(laptop_info)
            time.sleep(random.uniform(2, 5))  # Random delay between 2 and 5 seconds

        # Check if we've reached the last page
        next_button = driver.find_elements(By.CSS_SELECTOR, ".s-pagination-item.s-pagination-next")
        if not next_button or "a-disabled" in next_button[0].get_attribute("class"):
            print("Reached the last page. Stopping scraping.")
            driver.quit()
            break

        driver.quit()
        page += 1
        time.sleep(random.uniform(3, 7))  # Random delay between pages

    return laptops

def save_to_csv(laptops, filename='laptops.csv'):
    if not laptops:
        print("No data to save.")
        return

    keys = [
        "title", "price", "Standing screen display size", "Screen Resolution", 
        "Processor", "RAM", "Hard Drive", "Graphics Coprocessor", "Chipset Brand", 
        "Wireless Type", "Average Battery Life (in hours)", "Brand", "Series", 
        "Operating System", "Item Weight", "Color", "Processor Brand", 
        "Number of Processors", "Batteries"
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(laptops)

    print(f"Data saved to {filename}")

if __name__ == "__main__":
    base_url = "https://www.amazon.com/s?k=laptop"
    while True:
        try:
            num_pages = int(input("Enter the number of pages to scrape (each page contains up to 16 laptops): "))
            if num_pages > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    laptops = scrape_laptops(base_url, num_pages)
    save_to_csv(laptops)