from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import csv
import time
import os

def setup_driver():
    options = webdriver.EdgeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--log-level=3')  # fatal onlys
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    service = Service('C:/edgedriver_win64/msedgedriver.exe')  # Update this path
    
    if os.name == 'posix':  # for Mac/Linux
        return webdriver.Edge(service=service, options=options, service_args=['--quiet'])
    elif os.name == 'nt':  # for Windows
        return webdriver.Edge(service=service, options=options)

def extract_product_info(driver, url):
    driver.get(url)
    
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "productTitle")))
    except TimeoutException:
        print(f"Timeout waiting for page to load: {url}")
        return None

    product = {}
    
    # Extract title
    product['title'] = driver.find_element(By.ID, "productTitle").text.strip()
    
    # Extract price
    try:
        price_element = driver.find_element(By.CLASS_NAME, "a-price-whole")
        price_fraction = driver.find_element(By.CLASS_NAME, "a-price-fraction")
        product['price'] = f"{price_element.text}.{price_fraction.text}"
    except NoSuchElementException:
        product['price'] = "N/A"
    
    # Extract details from the product details section
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

def scrape_laptops(base_url, num_pages):
    driver = setup_driver()
    laptops = []
    page = 1

    while page <= num_pages:
        url = f"{base_url}&page={page}"
        driver.get(url)
        print(f"Scraping page {page}")
        
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.s-main-slot")))
        except TimeoutException:
            print(f"Timeout waiting for page {page} to load")
            page += 1
            continue

        laptop_links = driver.find_elements(By.CSS_SELECTOR, "a.a-link-normal.s-underline-text.s-underline-link-text.s-link-style.a-text-normal")
        links = [link.get_attribute('href') for link in laptop_links if link.get_attribute('href')]

        for i, link in enumerate(links, 1):
            print(f"Scraping laptop {i} of {len(links)} on page {page}")
            laptop_info = extract_product_info(driver, link)
            if laptop_info:
                laptops.append(laptop_info)
            time.sleep(2)  # Delay to be respectful to the website

        page += 1
        time.sleep(3)  # Additional delay between pages

        # Check if we've reached the last page
        next_button = driver.find_elements(By.CSS_SELECTOR, ".s-pagination-item.s-pagination-next")
        if not next_button or "a-disabled" in next_button[0].get_attribute("class"):
            print("Reached the last page. Stopping scraping.")
            break

    driver.quit()
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
