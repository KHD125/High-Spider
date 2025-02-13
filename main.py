# Global Scrapy settings – these variables are used by scrapy.cfg and setup.py.
BOT_NAME = 'high_spider'
SPIDER_MODULES = ['main']
NEWSPIDER_MODULE = 'main'
ROBOTSTXT_OBEY = False  # We ignore robots.txt because we're scraping a search engine.
LOG_LEVEL = 'INFO'

import scrapy
from urllib.parse import urljoin

class BingSearchSpider(scrapy.Spider):
    name = "bing_search_spider"
    
    def __init__(self, keyword="08041020", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build the Bing query: site:https://gst.jamku.app/gstin/ "08041020"
        query = f'site:https://gst.jamku.app/gstin/ "{keyword}"'
        # Replace spaces with + and encode quotes as %22
        self.start_urls = [
            "https://www.bing.com/search?q=" + query.replace(" ", "+").replace('"', "%22")
        ]

    def parse(self, response):
        # Each Bing result is usually in a <li class="b_algo"> element.
        results = response.css("li.b_algo")
        for result in results:
            url = result.css("h2 a::attr(href)").get()
            if url:
                yield {"url": url}
        # Follow the "Next" page, if available.
        next_page = response.css("a.sb_pagN::attr(href)").get()
        if next_page:
            yield scrapy.Request(urljoin(response.url, next_page), callback=self.parse)

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
    process.crawl(BingSearchSpider, keyword="08041020")
    process.start()
