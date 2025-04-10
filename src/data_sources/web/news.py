#!/usr/bin/env python3
# news_headline_extractor.py - Location-based News Headline Extractor

import asyncio
import aiohttp
import logging
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import re
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
HTTP_TIMEOUT = 15
SCAN_INTERVAL = 60  # seconds between scans
MAX_RESULTS_PER_SEARCH = 20

@dataclass
class NewsHeadline:
    """Represents a news headline with essential metadata."""
    title: str
    source: str
    url: str
    published_date: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    location: Optional[str] = None

class NewsAPIClient:
    """Client for accessing news APIs."""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        
        # The free tier of NewsAPI requires an API key and has limitations
        # Get your API key at https://newsapi.org/
        self.newsapi_key = "c7b2727ab89d4ce1a091d4934fdf4171"  # Add your NewsAPI key here
        
        # The free tier of GNews also requires an API key
        # Get your API key at https://gnews.io/
        self.gnews_key = ""    # Add your GNews API key here
        
        # MediaStack API Key
        # Get your API key at https://mediastack.com/
        self.mediastack_key = ""  # Add your MediaStack API key here
        
    async def search_newsapi(self, query: str, language: str = "en") -> List[NewsHeadline]:
        """Search using NewsAPI."""
        if not self.newsapi_key:
            logger.warning("NewsAPI key not configured. Skipping NewsAPI search.")
            return []
            
        # Use two different endpoints to maximize results:
        # 1. First try top-headlines which gives breaking news
        top_headlines = await self._search_newsapi_top_headlines(query, language)
        
        # 2. Then try everything endpoint for more comprehensive results
        everything_headlines = await self._search_newsapi_everything(query, language)
        
        # Combine results
        all_headlines = top_headlines + everything_headlines
        
        # Remove duplicates (same title and URL)
        seen_keys = set()
        unique_headlines = []
        
        for headline in all_headlines:
            key = f"{headline.title}|{headline.url}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique_headlines.append(headline)
                
        logger.info(f"Found {len(unique_headlines)} unique headlines from NewsAPI")
        return unique_headlines
        
    async def _search_newsapi_top_headlines(self, query: str, language: str = "en") -> List[NewsHeadline]:
        """Search using NewsAPI top-headlines endpoint."""
        url = "https://newsapi.org/v2/top-headlines"
        
        # For top headlines, we need to use the 'q' parameter more selectively
        # Extract location as a potential country code
        country_code = None
        if query.lower() in ["us", "usa", "united states"]:
            country_code = "us"
        elif query.lower() in ["uk", "united kingdom"]:
            country_code = "gb"
        elif query.lower() in ["india"]:
            country_code = "in"
        # Add more country mappings as needed
        
        params = {
            "q": query,
            "language": language,
            "apiKey": self.newsapi_key,
            "pageSize": MAX_RESULTS_PER_SEARCH
        }
        
        # Add country parameter if we have a valid country code
        if country_code:
            params["country"] = country_code
            # If we're using country, we can make the query more general
            if "q" in params:
                del params["q"]
        
        try:
            async with self.session.get(url, params=params, timeout=HTTP_TIMEOUT) as response:
                if response.status != 200:
                    logger.error(f"NewsAPI top-headlines error: {response.status} - {await response.text()}")
                    return []
                    
                data = await response.json()
                headlines = []
                
                for article in data.get("articles", []):
                    headlines.append(NewsHeadline(
                        title=article.get("title", ""),
                        source=article.get("source", {}).get("name", "NewsAPI"),
                        url=article.get("url", ""),
                        published_date=article.get("publishedAt", ""),
                        description=article.get("description", ""),
                        category=None,  # NewsAPI doesn't provide category
                        location=None  # Extract from context if possible
                    ))
                
                logger.debug(f"Found {len(headlines)} headlines from NewsAPI top-headlines")
                return headlines
                
        except Exception as e:
            logger.error(f"Error searching NewsAPI top-headlines: {e}")
            return []
    
    async def _search_newsapi_everything(self, query: str, language: str = "en") -> List[NewsHeadline]:
        """Search using NewsAPI everything endpoint."""
        url = "https://newsapi.org/v2/everything"
        
        # Get current date in UTC
        now = datetime.utcnow()
        
        # Format as ISO string and trim to just the date portion
        today = now.strftime("%Y-%m-%d")
        
        # Calculate yesterday's date
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        
        params = {
            "q": query,
            "language": language,
            "sortBy": "publishedAt",
            "apiKey": self.newsapi_key,
            "pageSize": MAX_RESULTS_PER_SEARCH,
            "from": yesterday,  # Only get news from the last 24 hours
            "to": today
        }
        
        try:
            async with self.session.get(url, params=params, timeout=HTTP_TIMEOUT) as response:
                if response.status != 200:
                    logger.error(f"NewsAPI error: {response.status} - {await response.text()}")
                    return []
                    
                data = await response.json()
                headlines = []
                
                for article in data.get("articles", []):
                    headlines.append(NewsHeadline(
                        title=article.get("title", ""),
                        source=article.get("source", {}).get("name", "NewsAPI"),
                        url=article.get("url", ""),
                        published_date=article.get("publishedAt", ""),
                        description=article.get("description", ""),
                        category=None,  # NewsAPI doesn't provide category
                        location=None  # Extract from context if possible
                    ))
                
                logger.info(f"Found {len(headlines)} headlines from NewsAPI")
                return headlines
                
        except Exception as e:
            logger.error(f"Error searching NewsAPI: {e}")
            return []
    
    async def search_gnews(self, query: str, language: str = "en") -> List[NewsHeadline]:
        """Search using GNews."""
        if not self.gnews_key:
            logger.warning("GNews API key not configured. Skipping GNews search.")
            return []
            
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "lang": language,
            "sortby": "publishedAt",
            "apikey": self.gnews_key,
            "max": MAX_RESULTS_PER_SEARCH
        }
        
        try:
            async with self.session.get(url, params=params, timeout=HTTP_TIMEOUT) as response:
                if response.status != 200:
                    logger.error(f"GNews API error: {response.status} - {await response.text()}")
                    return []
                    
                data = await response.json()
                headlines = []
                
                for article in data.get("articles", []):
                    headlines.append(NewsHeadline(
                        title=article.get("title", ""),
                        source=article.get("source", {}).get("name", "GNews"),
                        url=article.get("url", ""),
                        published_date=article.get("publishedAt", ""),
                        description=article.get("description", ""),
                        category=None,
                        location=None
                    ))
                
                logger.info(f"Found {len(headlines)} headlines from GNews")
                return headlines
                
        except Exception as e:
            logger.error(f"Error searching GNews: {e}")
            return []
    
    async def search_mediastack(self, query: str, languages: str = "en") -> List[NewsHeadline]:
        """Search using MediaStack API."""
        if not self.mediastack_key:
            logger.warning("MediaStack API key not configured. Skipping MediaStack search.")
            return []
            
        url = "http://api.mediastack.com/v1/news"
        params = {
            "access_key": self.mediastack_key,
            "keywords": query,
            "languages": languages,
            "sort": "published_desc",
            "limit": MAX_RESULTS_PER_SEARCH
        }
        
        try:
            async with self.session.get(url, params=params, timeout=HTTP_TIMEOUT) as response:
                if response.status != 200:
                    logger.error(f"MediaStack API error: {response.status} - {await response.text()}")
                    return []
                    
                data = await response.json()
                headlines = []
                
                for article in data.get("data", []):
                    headlines.append(NewsHeadline(
                        title=article.get("title", ""),
                        source=article.get("source", "MediaStack"),
                        url=article.get("url", ""),
                        published_date=article.get("published_at", ""),
                        description=article.get("description", ""),
                        category=article.get("category", None),
                        location=article.get("country", None)
                    ))
                
                logger.info(f"Found {len(headlines)} headlines from MediaStack")
                return headlines
                
        except Exception as e:
            logger.error(f"Error searching MediaStack: {e}")
            return []

    async def search_bing_news(self, query: str, market: str = "en-US", count: int = MAX_RESULTS_PER_SEARCH) -> List[NewsHeadline]:
        """Search using Bing News Search API."""
        # Get your API key at https://www.microsoft.com/en-us/bing/apis/bing-news-search-api
        bing_api_key = ""  # Add your Bing News Search API key here
        
        if not bing_api_key:
            logger.warning("Bing News Search API key not configured. Skipping Bing search.")
            return []
            
        url = "https://api.bing.microsoft.com/v7.0/news/search"
        headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
        params = {
            "q": query,
            "mkt": market,
            "count": count,
            "freshness": "Day"  # Options: Day, Week, Month
        }
        
        try:
            async with self.session.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT) as response:
                if response.status != 200:
                    logger.error(f"Bing News API error: {response.status} - {await response.text()}")
                    return []
                    
                data = await response.json()
                headlines = []
                
                for article in data.get("value", []):
                    headlines.append(NewsHeadline(
                        title=article.get("name", ""),
                        source=article.get("provider", [{}])[0].get("name", "Bing News"),
                        url=article.get("url", ""),
                        published_date=article.get("datePublished", ""),
                        description=article.get("description", ""),
                        category=article.get("category", None),
                        location=None  # Extract from context if needed
                    ))
                
                logger.info(f"Found {len(headlines)} headlines from Bing News")
                return headlines
                
        except Exception as e:
            logger.error(f"Error searching Bing News API: {e}")
            return []

    async def search_all(self, query: str, language: str = "en") -> List[NewsHeadline]:
        """Search across all configured news APIs."""
        tasks = []
        
        # Add API search tasks based on which keys are configured
        if self.newsapi_key:
            tasks.append(self.search_newsapi(query, language))
        if self.gnews_key:
            tasks.append(self.search_gnews(query, language))
        if self.mediastack_key:
            tasks.append(self.search_mediastack(query, language))
            
        # If no API keys are configured, log a warning
        if not tasks:
            logger.warning("No API keys configured. Using fallback methods.")
            # You could add fallback methods here like RSS feed parsing
            
        # Execute all search tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results, handle any exceptions
        all_headlines = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"API search error: {result}")
            else:
                all_headlines.extend(result)
                
        # Deduplicate headlines
        unique_headlines = {}
        for headline in all_headlines:
            # Use title+source as a simple deduplication key
            key = f"{headline.title}|{headline.source}"
            unique_headlines[key] = headline
            
        return list(unique_headlines.values())

class LocationBasedNewsExtractor:
    """Class for extracting news headlines based on location."""
    
    def __init__(self):
        """Initialize the news extractor."""
        self.session = None
        self.api_client = None
        self.seen_headlines = set()  # Set to track already seen headlines
        self.demo_mode = False  # Enable to simulate new headlines for testing
        self.demo_headline_index = 0  # Counter for demo headlines
        
    async def initialize(self):
        """Initialize HTTP session and API client."""
        # Configure aiohttp session
        connector = aiohttp.TCPConnector(limit=10, ssl=False)
        self.session = aiohttp.ClientSession(
            connector=connector,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8'
            }
        )
        self.api_client = NewsAPIClient(self.session)
        logger.info("News extractor initialized")
        
    async def search_news_for_location(self, location: str, additional_keywords: str = "", language: str = "en") -> List[NewsHeadline]:
        """Search for news related to a specific location."""
        query = location
        if additional_keywords:
            query = f"{location} {additional_keywords}"
            
        logger.info(f"Searching for news about: {query}")
        
        if self.demo_mode:
            # In demo mode, generate some simulated headlines for testing
            demo_headlines = self._generate_demo_headlines(location, additional_keywords)
            logger.info(f"DEMO MODE: Generated {len(demo_headlines)} test headlines")
            return demo_headlines
        
        # Normal mode - search APIs
        headlines = await self.api_client.search_all(query, language)
        
        # Additional filtering to ensure location relevance
        filtered_headlines = []
        location_lower = location.lower()
        
        for headline in headlines:
            title_lower = headline.title.lower()
            desc_lower = headline.description.lower() if headline.description else ""
            
            # Check if location is mentioned in title or description
            if location_lower in title_lower or location_lower in desc_lower:
                filtered_headlines.append(headline)
                
        logger.info(f"Found {len(filtered_headlines)} headlines relevant to {location}")
        return filtered_headlines
        
    def _generate_demo_headlines(self, location: str, category: str = "") -> List[NewsHeadline]:
        """Generate demo headlines for testing purposes."""
        self.demo_headline_index += 1
        current_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Template headlines we'll customize with location and category
        templates = [
            "Breaking: New development in {location} attracts attention",
            "Local government in {location} announces new initiative",
            "{location} residents react to recent {category} changes",
            "Expert analysis: What's happening in {location} {category} scene",
            "Survey shows {location} leads in {category} innovation",
            "International partnership brings new {category} opportunities to {location}",
            "Community event in {location} showcases {category} achievements",
            "Report: {location}'s {category} sector growing rapidly",
            "New study reveals interesting trends in {location} {category}",
            "Business leaders discuss {location}'s future in {category}"
        ]
        
        # News sources for demo
        sources = ["DemoNews", "TestDaily", "SampleTimes", "ExamplePost", "MockMedia"]
        
        # Create 1-3 new headlines each time
        num_headlines = min(3, len(templates))
        headlines = []
        
        for i in range(num_headlines):
            # Use the counter to select different headlines each time
            index = (self.demo_headline_index + i) % len(templates)
            
            # Fill in the template
            if category:
                title = templates[index].format(location=location, category=category)
            else:
                # If no category, replace {category} with "local"
                title = templates[index].format(location=location, category="local")
            
            # Create a unique URL
            url = f"https://demo-news.example.com/{location.lower().replace(' ', '-')}/{self.demo_headline_index + i}"
            
            # Select a source
            source = sources[i % len(sources)]
            
            # Create description
            description = f"This is a simulated news article about {location}"
            if category:
                description += f" focusing on {category}"
            description += ". Generated for testing purposes."
            
            headlines.append(NewsHeadline(
                title=title,
                source=source,
                url=url,
                published_date=current_time,
                description=description,
                category=category if category else "general",
                location=location
            ))
            
        return headlines
    
    async def start_monitoring(self, locations: List[str], categories: List[str] = None, interval: int = SCAN_INTERVAL):
        """Start monitoring for news about specific locations."""
        if not self.session:
            await self.initialize()
            
        logger.info(f"Starting news monitoring for locations: {', '.join(locations)}")
        
        # For better search results, create search queries that use time specifiers
        # This helps find newer content each scan
        time_specifiers = [
            "today", "latest", "breaking", "recent", "just in", 
            "new", "latest", "update"
        ]
        time_specifier_index = 0
        
        try:
            # First scan - display all initial results
            first_scan = True
            
            while True:
                all_results = []
                
                # Add a time specifier to search for newer content
                current_time_specifier = time_specifiers[time_specifier_index % len(time_specifiers)]
                time_specifier_index += 1
                
                # Search for each location
                for location in locations:
                    # Also search with categories if provided
                    if categories:
                        for category in categories:
                            # If demo mode, we don't need to add time specifiers
                            if not self.demo_mode:
                                search_query = f"{category} {current_time_specifier}"
                            else:
                                search_query = category
                                
                            results = await self.search_news_for_location(location, search_query)
                            all_results.extend(results)
                    else:
                        # If no categories specified, try using time specifier directly
                        if not self.demo_mode:
                            search_query = current_time_specifier
                        else:
                            search_query = ""
                            
                        results = await self.search_news_for_location(location, search_query)
                        all_results.extend(results)
                
                # Logs to help debugging
                logger.debug(f"Found {len(all_results)} total headlines before filtering")
                
                # Filter out previously seen headlines
                new_headlines = self._filter_new_headlines(all_results, first_scan)
                
                # Output only new headlines
                if new_headlines:
                    if not first_scan:
                        logger.info(f"Found {len(new_headlines)} new headlines!")
                    self._output_headlines(new_headlines)
                else:
                    if not first_scan:
                        logger.info("No new headlines found in this scan.")
                
                # After first scan, switch to incremental mode
                first_scan = False
                
                logger.info(f"Scanning for updates every {interval} seconds. Press Ctrl+C to exit...")
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring task cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            if self.session:
                await self.session.close()
                logger.info("HTTP session closed")
    
    def _get_headline_key(self, headline: NewsHeadline) -> str:
        """Generate a unique key for a headline to track seen status."""
        # Combine title and URL for uniqueness
        return f"{headline.title}|{headline.url}"
    
    def _filter_new_headlines(self, headlines: List[NewsHeadline], include_all: bool = False) -> List[NewsHeadline]:
        """Filter headlines to only include previously unseen ones."""
        new_headlines = []
        
        for headline in headlines:
            headline_key = self._get_headline_key(headline)
            
            # If this is the first scan or we haven't seen this headline before
            if include_all or headline_key not in self.seen_headlines:
                new_headlines.append(headline)
                # Add to seen headlines set
                self.seen_headlines.add(headline_key)
        
        return new_headlines
    
    def _output_headlines(self, headlines: List[NewsHeadline]):
        """Output headlines to console."""
        if not headlines:
            return
            
        logger.info(f"--- {len(headlines)} Headlines {'Found' if len(headlines) > 1 else 'Found'} ---")
        
        # Sort by published date (newest first)
        try:
            # Simple sort based on string comparison of dates
            sorted_headlines = sorted(
                headlines, 
                key=lambda h: h.published_date or "", 
                reverse=True
            )
        except Exception:
            # Fallback if sorting fails
            sorted_headlines = headlines
            
        # Print headlines
        for i, headline in enumerate(sorted_headlines, 1):
            logger.info(f"{i}. {headline.title}")
            logger.info(f"   Source: {headline.source} | Date: {headline.published_date or 'Unknown'}")
            if headline.description:
                desc = headline.description[:100] + ("..." if len(headline.description) > 100 else "")
                logger.info(f"   Description: {desc}")
            logger.info(f"   URL: {headline.url}")
            logger.info("")

async def main():
    """Main entry point for the news extractor."""
    print("\n===== LOCATION-BASED NEWS HEADLINE EXTRACTOR =====\n")
    print("This tool monitors news sources and displays only new headlines about your topics of interest.\n")
    
    # Get user input for locations to monitor
    print("Enter locations to monitor for news (comma-separated):")
    location_input = input("> ")
    locations = [loc.strip() for loc in location_input.split(",") if loc.strip()]
    
    if not locations:
        print("No locations specified. Using 'World News' as default.")
        locations = ["World News"]
        
    # Get optional categories
    print("\nEnter categories to focus on (comma-separated, or press Enter to skip):")
    print("Examples: politics, business, technology, sports, entertainment")
    category_input = input("> ")
    categories = [cat.strip() for cat in category_input.split(",") if cat.strip()]
    
    # Get scan interval
    print("\nEnter scan interval in seconds (default: 60):")
    interval_input = input("> ")
    try:
        interval = int(interval_input) if interval_input.strip() else SCAN_INTERVAL
    except ValueError:
        print(f"Invalid interval. Using default: {SCAN_INTERVAL} seconds")
        interval = SCAN_INTERVAL
    
    # Ask if user wants to enable demo mode
    print("\nEnable demo mode? (Generates test headlines for debugging) [y/N]:")
    demo_input = input("> ").strip().lower()
    demo_mode = demo_input in ['y', 'yes', 'true', '1']
        
    print(f"\nStarting news monitoring for: {', '.join(locations)}")
    if categories:
        print(f"Categories: {', '.join(categories)}")
    print(f"Scan interval: {interval} seconds")
    if demo_mode:
        print("DEMO MODE ENABLED: Will generate sample headlines for testing")
    print("\nInitial scan will show all matching headlines.")
    print("Subsequent scans will only show NEW headlines as they appear.")
    print("Press Ctrl+C at any time to exit.\n")
    
    extractor = LocationBasedNewsExtractor()
    extractor.demo_mode = demo_mode
    
    try:
        await extractor.initialize()
        await extractor.start_monitoring(locations, categories, interval)
    except KeyboardInterrupt:
        print("\nStopping news extractor (Ctrl+C pressed)")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nNews extractor stopped")

if __name__ == "__main__":
    asyncio.run(main())