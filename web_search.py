"""Small structured web search; no page crawling or knowledge indexing."""
import html
import logging
import os
import httpx
from urllib.parse import urlsplit, urlunsplit
from urllib.request import getproxies

from ddgs import DDGS

logger = logging.getLogger(__name__)
MAX_WEB_RESULTS = 3


def web_search(question):
    try:
        provider = os.getenv('WEB_SEARCH_PROVIDER', 'ddgs')
        if provider == 'tavily':
            key = os.getenv('TAVILY_API_KEY')
            if not key:
                logger.warning('TAVILY_API_KEY is not configured')
                return []
            response = httpx.post(
                'https://api.tavily.com/search', headers={'Authorization':'Bearer '+key},
                json={'query':question, 'max_results':MAX_WEB_RESULTS, 'search_depth':'basic',
                      'include_answer':False, 'include_raw_content':False}, timeout=15,
                trust_env=os.getenv('TAVILY_TRUST_ENV', 'true').lower() in {'true', '1', 'yes'}
            )
            response.raise_for_status()
            raw = response.json()['results']
        elif provider == 'ddgs':
            # Reuse existing proxy settings; never change system configuration.
            proxy = os.getenv('DDGS_PROXY') or getproxies().get('https') or getproxies().get('http')
            raw = DDGS(proxy=proxy, timeout=10).text(
                question, max_results=MAX_WEB_RESULTS,
                backend=os.getenv('WEB_SEARCH_BACKEND', 'auto'), safesearch='moderate'
            )
        else:
            logger.warning('Unsupported WEB_SEARCH_PROVIDER')
            return []
    except Exception:
        logger.exception('Web search unavailable')
        return []
    results = []
    seen = set()
    if not isinstance(raw, list):
        logger.warning('Invalid search response structure')
        return []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = html.unescape(str(item.get('title') or '')).strip()
        snippet = html.unescape(str(item.get('body') or item.get('content') or '')).strip()
        url = str(item.get('href') or item.get('url') or '').strip()
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
                continue
            url = urlunsplit(parsed._replace(fragment=''))
        except ValueError:
            continue
        if not title or not snippet or url in seen:
            continue
        seen.add(url)
        results.append({'title':title[:300], 'url':url, 'snippet':snippet[:1500]})
        if len(results) == MAX_WEB_RESULTS:
            break
    return results
