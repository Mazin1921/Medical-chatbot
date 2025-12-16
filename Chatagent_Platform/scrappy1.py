import requests
from bs4 import BeautifulSoup
import csv
from time import sleep
from urllib.parse import urljoin, urlparse

def is_valid_url(url):
    """
    Ensure the URL is from medlineplus.gov and not external
    """
    if not url:
        return False

    invalid = ['facebook.com', 'instagram.com', 'linkedin.com', 'youtube.com',
               'mailto:', 'tel:', 'twitter.com', 'pdf']
    
    parsed = urlparse(url)
    return ('medlineplus.gov' in parsed.netloc and 
            not any(x in url for x in invalid))

def clean_text(text):
    """
    Clean extra whitespace and line breaks
    """
    if not text:
        return ""
    return ' '.join(text.split())

def scrape_page(url):
    """
    Scrape title, headings, paragraphs, and extract valid internal links
    """
    print(f"🔍 Crawling: {url}")
    sleep(1.5)
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')

        # Clean page
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()

        title = clean_text(soup.title.string if soup.title else "")
        headings = [clean_text(h.get_text()) for h in soup.find_all(['h1', 'h2', 'h3'])]
        paragraphs = [clean_text(p.get_text()) for p in soup.find_all('p') if clean_text(p.get_text())]

        # Collect internal links
        links = []
        for a in soup.find_all('a', href=True):
            full_url = urljoin(url, a['href'])
            if is_valid_url(full_url):
                links.append(full_url)

        return {
            'url': url,
            'title': title,
            'headings': headings,
            'paragraphs': paragraphs,
            'links': list(set(links))  # de-duplicate
        }

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

def save_to_csv(records, filename):
    """
    Save data to a CSV file
    """
    keys = ['url', 'title', 'headings', 'paragraphs']
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({
                'url': record['url'],
                'title': record['title'],
                'headings': ' | '.join(record['headings']),
                'paragraphs': ' | '.join(record['paragraphs'])
            })

def main():
    # Phase 1: Start with main index pages
    start_urls = [
        "https://medlineplus.gov/encyclopedia.html",
        "https://medlineplus.gov/healthtopics.html"
    ]

    visited = set()
    queue = list(start_urls)
    collected_data = []

    # Phase 2: Visit each page (main + discovered internal links)
    while queue and len(visited) < 500 :  # Limit for safety
        current_url = queue.pop(0)
        if current_url in visited:
            continue

        page_data = scrape_page(current_url)
        visited.add(current_url)

        if page_data:
            collected_data.append(page_data)

            # Add new internal links for future scraping (depth 1)
            for link in page_data['links']:
                if link not in visited and link not in queue:
                    queue.append(link)

    save_to_csv(collected_data, "medical_content1.csv")
    print(f"\n✅ Scraping finished. Total pages saved: {len(collected_data)}")

if __name__ == "__main__":
    main()
