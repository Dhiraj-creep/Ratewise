"""Parse a locally saved Amazon product page. Never fetches or bypasses access controls."""
import re
from bs4 import BeautifulSoup
from .data import amazon_identity, MARKETS

def parse_saved_html(content: bytes, url: str):
    market, asin, canonical = amazon_identity(url)
    soup = BeautifulSoup(content, "html.parser")
    page_text = soup.get_text(" ", strip=True)
    if soup.select_one("#captchacharacters") or re.search(r"enter the characters you see below|sorry, we just need to make sure you're not a robot", page_text, re.I):
        raise ValueError("This is an access challenge, not a product page. Open Amazon normally and collect only accessible pages.")
    title = soup.select_one("#productTitle")
    if not title:
        raise ValueError("Product title not found. Use a saved product detail page, or manual capture.")
    page_asin = soup.select_one('input#ASIN')
    if page_asin and page_asin.get("value", "").upper() != asin:
        raise ValueError("The saved HTML product ASIN does not match the URL.")
    def text(selector):
        el = soup.select_one(selector)
        return el.get_text(" ", strip=True) if el else ""
    def value(selector, pattern=r"[\d,]+(?:\.\d+)?"):
        m = re.search(pattern, text(selector))
        return m.group(0).replace(",", "") if m else ""
    category = " > ".join(a.get_text(" ", strip=True) for a in soup.select("#wayfinding-breadcrumbs_feature_div li a"))
    rating_text = text("#acrPopover") or text("#averageCustomerReviews .a-icon-alt")
    rating_match = re.search(r"([1-5](?:\.\d+)?)\s+out of\s+5", rating_text, re.I)
    count_text = text("#acrCustomerReviewText")
    # Amazon commonly labels this as ratings; do not mislabel it as written reviews.
    count_match = re.fullmatch(r"\s*([\d,]+)\s+(?:global\s+)?ratings?\s*", count_text, re.I)
    review_match = re.fullmatch(r"\s*([\d,]+)\s+(?:customer\s+)?reviews?\s*", count_text, re.I)
    row = dict(asin=asin, source_url=canonical, currency=MARKETS[market],
               title=title.get_text(" ", strip=True), brand="", category=category,
               price=value("#corePriceDisplay_desktop_feature_div .a-price:not(.a-text-price) .a-offscreen, #corePrice_feature_div .a-price:not(.a-text-price) .a-offscreen"),
               list_price=value("#corePriceDisplay_desktop_feature_div .a-text-price .a-offscreen"),
               discount_pct="", rating=rating_match.group(1) if rating_match else "",
               rating_count=count_match.group(1).replace(",", "") if count_match else "",
               review_count=review_match.group(1).replace(",", "") if review_match else "",
               parent_asin="", collection_method="saved_amazon_html")
    warnings = ["Verify every extracted field against the displayed page before saving.",
                "Choose a consistent broad category; the extracted breadcrumb may be too specific.",
                "Brand, parent ASIN and written-review count may require manual entry."]
    return row, warnings
