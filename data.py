import re
import os
import random
import pickle
import asyncio
import aiohttp
import pandas as pd
from typing import Any
from typing import Optional
from bs4 import BeautifulSoup
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse


MAX_CONCURRENT_REQUESTS = 8
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def save(obj: Any, fn: str) -> None:
  with open(fn, 'wb') as f:
    pickle.dump(obj, f)

def load(fn: str) -> Any:
  with open(fn, 'rb') as f:
    return pickle.load(f)

@dataclass(frozen=True)
class Link:
  url: str
  text: str
  domain: str

@dataclass(frozen=True)
class Article:
  title: str
  text: str

async def _get_links(url: str) -> list[Link]:
  async with aiohttp.ClientSession() as session:
    try:
      async with session.get(url) as res:
        res.raise_for_status()
        html = await res.text()
    except aiohttp.ClientError as e:
      print(f"Error downloading webpage: {e}")
      return []
    
  soup = BeautifulSoup(html, 'html.parser')
  links = soup.find_all('a', href=True)
  a = []
  for x in links:
    href = x['href']
    full_url = urljoin(url, href)
    text = x.text.strip()
    domain = urlparse(full_url).netloc
    a.append(Link(url=full_url, text=text, domain=domain))
  return a

async def _get_article(link: Link) -> Optional[Article]:
  async with semaphore:
    await asyncio.sleep(random.uniform(0, 5))
    async with aiohttp.ClientSession() as session:
      try:
        async with session.get(link.url) as res:
          res.raise_for_status()
          html = await res.text()
      except aiohttp.ClientError as e:
        print(f"Error downloading the webpage: {e}")
        return None
    
  soup = BeautifulSoup(html, 'html.parser')
  
  for script_or_style in soup(['script', 'style']): script_or_style.extract()

  text = soup.get_text(separator='\n', strip=True)
  lines = (line.strip() for line in text.splitlines())
  text = '\n'.join(line for line in lines if line)
  text = text.split('Digg')[0].strip()
  return Article(title=link.text, text=text)

async def get_articles() -> pd.DataFrame:
  fn = 'articles.parquet'
  if os.path.exists(fn): return pd.read_parquet(fn)
  url = 'https://thelastpsychiatrist.com/archives.html'
  links = await _get_links(url)
  p = r'^https://thelastpsychiatrist\.com/\d{4}/\d{2}/[a-zA-Z0-9_]+\.html$'
  links = [x for x in links if re.match(p, x.url)]
  # filter
  tasks = [_get_article(x) for x in links]
  articles = [x for x in await asyncio.gather(*tasks) if x is not None]
  df = pd.DataFrame([{'title': x.title, 'text': x.text} for x in articles])
  df.to_parquet(fn)
  return df

if __name__ == '__main__':
  df = asyncio.run(get_articles())
  print(df.head())
