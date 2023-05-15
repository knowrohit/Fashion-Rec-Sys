import requests
from bs4 import BeautifulSoup
import feedparser

def get_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the main content container
    content = soup.find('div', {'class': 'content-and-rail'})
    if content:
        # Remove unwanted elements
        for script in content(['script', 'style', 'aside', 'figure']):
            script.decompose()

        # Extract text from paragraphs
        paragraphs = content.find_all('p')
        text = '\n'.join([p.get_text(strip=True) for p in paragraphs])

        return text
    return ''

def get_feed_articles(feed_url):
    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries:
        title = entry.title
        link = entry.link
        content = entry.summary
        full_content = get_article_content(link)
        articles.append(f"{title}\n\n{content}\n\n{full_content}\n{link}\n")
    return articles

def save_articles_to_file(filename, articles):
    with open(filename, 'w') as f:
        for article in articles:
            f.write(article + '\n\n')

if __name__ == '__main__':
    rss_feed_url = 'https://www.vogue.com/rss'
    articles = get_feed_articles(rss_feed_url)
    save_articles_to_file('vogue_articles.txt', articles)
