# Global Scrapy settings – these module-level variables are used by scrapy.cfg and setup.py.
BOT_NAME = 'high_spider'
SPIDER_MODULES = ['main']
NEWSPIDER_MODULE = 'main'
ROBOTSTXT_OBEY = False  # We're scraping a search engine; ignore robots.txt.
LOG_LEVEL = 'INFO'

import scrapy
from urllib.parse import urljoin

class BingSearchSpider(scrapy.Spider):
    name = "bing_search_spider"
    
    def __init__(self, keyword="08041020", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyword = keyword
        # Build the search query.
        # Example: site:https://gst.jamku.app/gstin/ "08041020"
        query = f'site:https://gst.jamku.app/gstin/ "{keyword}"'
        # Construct the Bing search URL.
        # Replace spaces with + and quotes with %22.
        self.start_urls = [
            "https://www.bing.com/search?q=" + query.replace(" ", "+").replace('"', "%22")
        ]
        self.logger.info("Starting URL: " + self.start_urls[0])
    
    def parse(self, response):
        self.logger.info("Parsing page: " + response.url)
        # Extract each Bing result; they are usually in <li class="b_algo">
        results = response.css("li.b_algo")
        self.logger.info(f"Found {len(results)} result items on this page.")
        for result in results:
            url = result.css("h2 a::attr(href)").get()
            if url and url.startswith("https://gst.jamku.app/gstin/"):
                # Yield only the URL.
                yield {"url": url}
        
        # Follow the "Next" page if available.
        next_page = response.css("a.sb_pagN::attr(href)").get()
        if not next_page:
            # Fallback selector if Bing changes the title attribute.
            next_page = response.css("a[title='Next page']::attr(href)").get()
        if next_page:
            next_url = urljoin(response.url, next_page)
            self.logger.info("Following next page: " + next_url)
            yield scrapy.Request(next_url, callback=self.parse)
        else:
            self.logger.info("No next page found.")

# Local testing block.
if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess
    process = CrawlerProcess(settings={
        'BOT_NAME': BOT_NAME,
        'SPIDER_MODULES': SPIDER_MODULES,
        'NEWSPIDER_MODULE': NEWSPIDER_MODULE,
        'ROBOTSTXT_OBEY': ROBOTSTXT_OBEY,
        'LOG_LEVEL': LOG_LEVEL,
        'FEEDS': {
            'output.json': {'format': 'json', 'overwrite': True},
        },
    })
    process.crawl(BingSearchSpider, keyword="08041020")
    process.start()
