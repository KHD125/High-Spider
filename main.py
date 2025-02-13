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
        # If the page contains the keyword, yield its URL.
        if self.keyword in response.text.lower():
            yield {"url": response.url}
        # Follow internal links (only within the allowed domain).
        for href in response.css("a::attr(href)").getall():
            absolute_url = urljoin(response.url, href)
            if absolute_url.startswith("https://gst.jamku.app"):
                yield scrapy.Request(absolute_url, callback=self.parse)

# This block is used for local testing.
if __name__ == "__main__":
    process = CrawlerProcess(settings={
        'BOT_NAME': 'high_spider',
        'ROBOTSTXT_OBEY': True,
        'LOG_LEVEL': 'INFO',
        # Optionally, you can define FEEDS to export data locally
        'FEEDS': {
            'output.json': {
                'format': 'json',
                'overwrite': True,
            },
        },
    })
    process.crawl(GstinSpider, keyword="08041020")
    process.start()
