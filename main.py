# Global Scrapy settings – these variables are used by scrapy.cfg and setup.py
BOT_NAME = 'high_spider'
SPIDER_MODULES = ['main']
NEWSPIDER_MODULE = 'main'
ROBOTSTXT_OBEY = False  # We're scraping search results, so we ignore robots.txt here.
LOG_LEVEL = 'INFO'

import scrapy
from urllib.parse import urljoin, urlparse

class SearchEngineSpider(scrapy.Spider):
    name = "search_spider"

    def __init__(self, keyword="08041020", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyword = keyword
        # Build the Bing search query.
        # Query: site:https://gst.jamku.app/gstin/ "08041020"
        query = f'site:https://gst.jamku.app/gstin/ "{keyword}"'
        # Construct Bing search URL (spaces replaced with +, quotes URL-encoded)
        self.start_urls = [
            "https://www.bing.com/search?q=" + query.replace(" ", "+").replace('"', "%22")
        ]

    def parse(self, response):
        # Parse Bing search results: each result is in a <li class="b_algo"> element.
        results = response.css("li.b_algo")
        for result in results:
            url = result.css("h2 a::attr(href)").get()
            if url and url.startswith("https://gst.jamku.app/gstin/"):
                # Extract the GSTIN number from the URL (last path component)
                gstin = urlparse(url).path.rstrip("/").split("/")[-1]
                yield {"url": url, "gstin": gstin}
        # Follow the "Next" page link if available.
        next_page = response.css("a.sb_pagN::attr(href)").get()
        if next_page:
            next_url = urljoin(response.url, next_page)
            yield scrapy.Request(next_url, callback=self.parse)

# For local testing only.
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
    process.crawl(SearchEngineSpider, keyword="08041020")
    process.start()
