# Global Scrapy settings – these are imported by Zyte
BOT_NAME = 'high_spider'
SPIDER_MODULES = ['main']
NEWSPIDER_MODULE = 'main'
ROBOTSTXT_OBEY = True
LOG_LEVEL = 'INFO'

import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urljoin

class GstinSpider(scrapy.Spider):
    name = "gstin_spider"
    allowed_domains = ["gst.jamku.app"]
    start_urls = ["https://gst.jamku.app/gstin/"]

    def __init__(self, keyword="08041020", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyword = keyword.lower()

    def parse(self, response):
        # Check if this page is likely a GSTIN details page
        if self.keyword in response.text.lower():
            # Extract GSTIN from URL (last path component)
            gstin = response.url.rstrip('/').split('/')[-1]
            # The following selectors are examples.
            # Inspect the page to adjust them to the correct elements.
            owner = response.css("div.owner::text").get(default="Not Found").strip()
            location = response.css("div.location::text").get(default="Not Found").strip()
            compliance = response.css("div.compliance::text").get(default="Not Found").strip()
            
            yield {
                "url": response.url,
                "gstin": gstin,
                "owner": owner,
                "location": location,
                "compliance": compliance,
            }
        # Follow internal links to crawl additional pages.
        for href in response.css("a::attr(href)").getall():
            absolute_url = urljoin(response.url, href)
            if absolute_url.startswith("https://gst.jamku.app"):
                yield scrapy.Request(absolute_url, callback=self.parse)

# For local testing only.
if __name__ == "__main__":
    process = CrawlerProcess(settings={
        'BOT_NAME': BOT_NAME,
        'SPIDER_MODULES': SPIDER_MODULES,
        'NEWSPIDER_MODULE': NEWSPIDER_MODULE,
        'ROBOTSTXT_OBEY': ROBOTSTXT_OBEY,
        'LOG_LEVEL': LOG_LEVEL,
        'FEEDS': {
            'output.json': {
                'format': 'json',
                'overwrite': True,
            },
        },
    })
    process.crawl(GstinSpider, keyword="08041020")
    process.start()
