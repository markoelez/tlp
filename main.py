import os
import asyncio
import pandas as pd
from data import get_articles
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


OPENAI_API_TOKEN = os.environ['OPENAI_API_TOKEN']

embedding_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

def _chunk(row: pd.Series):
  dat = row['text'].split('\n')
  dat = [x for x in dat if len(x.split()) >= 5]
  blacklist = ['@', '|', '©', 'Subscribe to this blog', 'Movable Type']
  dat = [x for x in dat if all(v not in x for v in blacklist)]
  dat = ['\n'.join(dat[i:i+3]) for i in range(0, len(dat), 3)]
  return pd.DataFrame({
    'title': row['title'],
    'text': dat,
    'n_chunk': range(1, len(dat) + 1)
  })

def _generate_embeddings(df: pd.DataFrame, chunk: bool = False) -> pd.DataFrame:
  fn = 'embeddings.parquet'
  if os.path.exists(fn):
    df = pd.read_parquet(fn)
    df['embeddings'] = df['embeddings'].apply(lambda x: x.reshape(1, -1))
    return df
  if chunk: df = pd.concat(df.apply(_chunk, axis=1).to_list(), ignore_index=True)
  df['embeddings'] = df['text'].apply(lambda x: embedding_model.encode(x))
  df.to_parquet(fn)
  df['embeddings'] = df['embeddings'].apply(lambda x: x.reshape(1, -1))
  return df

async def _sample(messages: list[str], temperature: float = 1.0, max_len: int = 512) -> str:
  client = AsyncOpenAI(api_key=OPENAI_API_TOKEN)
  res = await client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=temperature,
    max_tokens=max_len)
  return res.choices[0].message.content.strip()

async def _respond(query: str, text: str) -> str:
  messages = [{'role': 'user', 'content': '''\
  Use the following Sources to respond to a user's search query.

  Do not reference any information not found in the Context!

  Search Query: {query}

  Context:

  {text}

  Remember: do not reference any information not found in the Context!
  
  Adhere to these instructions:
  * If you reference information in a source, cite its title.
  * Be extremely thorough and verbose to ensure you impart all relevant information.
  * Use multiple paragraphs.

  Return just your answer with no other commentary.'''.format(text=text, query=query)}]
  return await _sample(messages, temperature=0.5, max_len=4096)

def _fmt_context(results: pd.DataFrame) -> str:
  def _fmt(row: dict):
    title, text = row['title'], row['text']
    return '''
    title: {title}
    snippet: {text}
    ---
    '''.format(title=title, text=text)
  return '\n'.join([_fmt(x[1].to_dict()) for x in results.iterrows()])

async def _search(query: str, df: pd.DataFrame, k: int = 10) -> str:
  query_embedding = embedding_model.encode([query])
  df['similarity'] = df['embeddings'].apply(lambda x: cosine_similarity(query_embedding, x)[0][0])
  df = df.sort_values(by='similarity', ascending=False)
  return df.head(n=k)

async def main():
  df = await get_articles()
  df = _generate_embeddings(df, chunk=True)
  
  while True:
    
    query = input('query: ')

    rdf = await _search(query, df, k=20)

    text = _fmt_context(rdf)

    res = await _respond(query, text)

    print('*' * 100)
    print('*' * 100)
    print('*' * 100)
    
    print(res)

    print('*' * 100)
    print('*' * 100)
    print('*' * 100)

if __name__ == '__main__': asyncio.run(main())
