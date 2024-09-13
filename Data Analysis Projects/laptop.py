import time
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

# Set up Microsoft Edge WebDriver
edge_options = Options()
edge_options.use_chromium = True
edge_service = Service('C:/edgedriver_win64/msedgedriver.exe')
driver = webdriver.Edge(service=edge_service, options=edge_options)

# Function to scrape laptop details from a single page
def scrape_laptop_details(url):
    driver.get(url)
    time.sleep(2)  # Wait for the page to load

    # Wait for the main content to be present
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "container"))
    )

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Extract the title
    title = soup.find('h1').string.strip() if soup.find('h1') else 'N/A'

    # Function to safely extract text from <td> elements
    def find_td_value(label):
        td = soup.find('td', string=lambda text: text and label.lower() in text.lower())
        return td.find_next('td').text.strip() if td else 'N/A'

    # Function to extract specification table details more robustly
    def get_spec(soup, spec_name):
        # Locate the <thead> or <td> with the specification name
        thead_or_td = soup.find(lambda tag: tag.name in ['thead', 'td'] and spec_name.lower() in tag.get_text(strip=True).lower())
        
        # If found, extract the next attribute details within the <tbody>
        if thead_or_td:
            next_td = thead_or_td.find_next('td', class_='attribute-value')
            return next_td.get_text(strip=True) if next_td else 'N/A'
        return 'N/A'

    # Extract product details
    product_price = find_td_value('product price')
    stock_status = find_td_value('Stock Status')
    product_model = find_td_value('Product model')
    warranty = find_td_value('Warranty')

    # Extract Brand details
    brand = find_td_value('Brand')  # Assuming Brand is located in a different table.

    # Extract Processor, Memory, Graphics, Display and Battery details
    processor = f"{get_spec(soup, 'Processor Model')} {get_spec(soup, 'Processor Speed')}"
    memory = f"{get_spec(soup, 'Memory Size')} {get_spec(soup, 'Memory Type')}"
    display = f"{get_spec(soup, 'Screen Size')} {get_spec(soup, 'Resolution')}"
    graphics = f"{get_spec(soup, 'GPU Chipset')} {get_spec(soup, 'GPU Memory Size')}"
    battery = f"{get_spec(soup, 'Battery Type')} {get_spec(soup, 'Battery capacity')}"

    
    # Extract specifications
    # processor = get_spec('Processor Model')
    # memory = get_spec('Memory Size')
    storage = get_spec('Storage')
    # display = f"{get_spec('Screen Size')} {get_spec('Resolution')}"
    # graphics = get_spec('GPU Chipset')
    webcam = get_spec('WebCam')
    keyboard = get_spec('Keyboard')
    networking = f"WiFi: {get_spec('WiFi')}, Bluetooth: {get_spec('Bluetooth')}"
    adapter = get_spec('Adapter')
    # battery = get_spec('Battery capacity')
    color = get_spec('Color')
    weight = get_spec('Weight')
    brand = get_spec('Brand')
    operating_system = get_spec('Operating System')

    # Return the collected details as a dictionary
    return {
        'Title': title,
        'Product Price': product_price,
        'Stock Status': stock_status,
        'Product Model': product_model,
        'Warranty': warranty,
        'Processor': processor,
        'Memory': memory,
        'Storage': storage,
        'Display': display,
        'Graphics': graphics,
        'WebCam': webcam,
        'Keyboard': keyboard,
        'Networking': networking,
        'Adapter': adapter,
        'Battery': battery,
        'Color': color,
        'Weight': weight,
        'Brand': brand,
        'Operating System': operating_system
    }

# Main scraping loop
base_url = 'https://www.techlandbd.com/brand-laptops?page='
all_laptops = []

for page in range(1, 2):  # Adjust range based on the number of pages
    print(f"Scraping page {page}...")
    driver.get(f"{base_url}{page}")
    time.sleep(2)  # Wait for the page to load

    # Find all laptop links on the current page
    laptop_links = driver.find_elements(By.CSS_SELECTOR, "div.name a")
    laptop_urls = [link.get_attribute('href') for link in laptop_links]

    # Scrape details for each laptop
    for url in laptop_urls:
        laptop_details = scrape_laptop_details(url)
        all_laptops.append(laptop_details)
        time.sleep(1)  # Be polite to the server

# Create a pandas DataFrame and save to CSV
df = pd.DataFrame(all_laptops)
df.to_csv('laptop_data.csv', index=False)

# Close the browser
driver.quit()

print("Scraping completed. Data saved to laptop_data.csv")