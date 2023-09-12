import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from db_functions import process_item
from indexer import indexer
import os
import json
from urllib.parse import urljoin
from collections import deque
import time

N = 300
count = 0
start_url = "https://www.cse.ust.hk/~kwtleung/COMP4321/testpage.htm"
visited = set()

def scrape_page(start_url, N=300):
    count = 0
    visited = set()
    queue = deque([start_url])

    while queue and count < N:
        url = queue.popleft()

        # Check if the URL has already been visited
        if url in visited:
            continue

        # Visit the URL and extract the page content
        response = requests.get(url, verify=True)
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')

        # Extract URL
        url = str(response.url)

        # Check if the website has been fetched before
        if os.path.exists('page_info.json') and os.path.exists('url_to_pageid.json'):
            with open('page_info.json') as f:
                page_info = json.load(f)
            with open('url_to_pageid.json') as f:
                url_to_pageid = json.load(f)
            if url in url_to_pageid:
                if url_to_pageid[url] in page_info:
                    return

        # Extract Last Modified date from the header
        # Search for time tag with itemprop="dateModified" or itemprop="datePublished"
        time_tag = soup.find("time", attrs={"itemprop": ["dateModified", "datePublished"]})
        if time_tag:
            last_modified = time_tag["datetime"]
        elif "Last-Modified" in response.headers:
            last_modified = response.headers["Last-Modified"]
        else:
            last_modified = soup.find('meta', {'name': "date"})['content']
        
        # Extract title from the header (if present)
        title_tag = soup.title
        title = str(title_tag.string).replace('/n','').strip() if title_tag else ''
        # Calculate the size of the website
        if "Content-Length" in response.headers:
            download_size = int(response.headers["Content-Length"])
        else:
            download_size = str(len(response.content))
        # Extract all stripped text
        raw_text = soup.get_text().strip()
        # Lowercase and tokenize the text data into array
        text = raw_text.lower()
        tokenizer = RegexpTokenizer(r'[a-z]+')
        tokens = tokenizer.tokenize(text)
        # Extract child links
        link_set = set([a.get('href') for a in soup.find_all('a') if a.get('href')])
        child_links = [urljoin(url, link) for link in link_set]

        # Store the data into the database
        item = {
            "Page title": title,
            "URL": url,
            "Last modification date": last_modified,
            "Size of page": download_size,
            "Words": tokens,
            "Child links": child_links
        }
        process_item(item)
        print(f"Fetched \'{title}\'")
        # Mark the URL as visited
        visited.add(url)
        count += 1

        # Add child links to the queue for BFS traversal
        for link in child_links:
            queue.append(link)

if __name__ == '__main__':
    # Record the start time
    start_time = time.time()
    # Start the scraping process
    scrape_page(start_url)
    # Record the end time
    end_time = time.time()# Print the elapsed time
    # Calculate the elapsed time (52s)
    elapsed_time = end_time - start_time
    print("Spider Elapsed time: {:.2f} seconds".format(elapsed_time))
    # Record the start time
    start_time = time.time()
    indexer()
    # Record the end time
    end_time = time.time()# Print the elapsed time
    # Calculate the elapsed time (0.89s)
    elapsed_time = end_time - start_time
    print("Indexer Elapsed time: {:.2f} seconds".format(elapsed_time))
