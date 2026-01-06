"""
GNews API MCP Server - Best Practices Implementation

This server provides access to the GNews API through the Model Context Protocol (MCP).
It exposes two main tools for fetching news data:
1. gnews_search_news - Search for news articles with specific keywords
2. gnews_get_top_headlines - Get trending news articles by category

Features:
- Full support for GNews API parameters
- Comprehensive error handling with actionable messages
- Pydantic input validation with ConfigDict
- Support for JSON and Markdown response formats
- Complete tool annotations
- Follows all MCP best practices
"""

import os
import json
import logging
from typing import Optional, Literal, List
from enum import Enum

import httpx
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings


# Configure logging for STDIO transport (writes to stderr)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Create FastMCP server with transport security settings
mcp = FastMCP(
    name="gnews_mcp",
    instructions="A Model Context Protocol server for accessing GNews API. Provides tools to search news articles and get top headlines.",
    transport_security=TransportSecuritySettings(
        allowed_hosts=[
            "demo-gnews-remotemcp-auth-dv8h.onrender.com",
            "localhost",
            "127.0.0.1",
        ],
        allowed_origins=["*"],  # Allow all origins since we handle auth separately
    ),
)

# Supported languages and countries (from GNews API documentation)
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "zh": "Chinese",
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ml": "Malayalam",
    "mr": "Marathi",
    "no": "Norwegian",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "uk": "Ukrainian",
}

SUPPORTED_COUNTRIES = {
    "au": "Australia",
    "br": "Brazil",
    "ca": "Canada",
    "cn": "China",
    "eg": "Egypt",
    "fr": "France",
    "de": "Germany",
    "gr": "Greece",
    "hk": "Hong Kong",
    "in": "India",
    "ie": "Ireland",
    "it": "Italy",
    "jp": "Japan",
    "nl": "Netherlands",
    "no": "Norway",
    "pk": "Pakistan",
    "pe": "Peru",
    "ph": "Philippines",
    "pt": "Portugal",
    "ro": "Romania",
    "ru": "Russian Federation",
    "sg": "Singapore",
    "es": "Spain",
    "se": "Sweden",
    "ch": "Switzerland",
    "tw": "Taiwan",
    "ua": "Ukraine",
    "gb": "United Kingdom",
    "us": "United States",
}

CATEGORIES = [
    "general",
    "world",
    "nation",
    "business",
    "technology",
    "entertainment",
    "sports",
    "science",
    "health",
]


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


class NewsResponse(BaseModel):
    """Represents a news API response"""

    totalArticles: int
    articles: List[dict]


class SearchNewsInput(BaseModel):
    """Input model for news search operations."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    q: str = Field(
        ...,
        description="Search keywords. Use logical operators like AND, OR, NOT. Use quotes for exact phrases.",
        min_length=1,
        max_length=200,
    )
    lang: Optional[str] = Field(
        default=None,
        description=f"Language code (2 letters). Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}",
    )
    country: Optional[str] = Field(
        default=None,
        description=f"Country code (2 letters). Supported: {', '.join(SUPPORTED_COUNTRIES.keys())}",
    )
    max_articles: Optional[int] = Field(
        default=10, description="Number of articles to return (1-100)", ge=1, le=100
    )
    search_in: Optional[str] = Field(
        default=None,
        description="Search in specific fields: title, description, content (comma-separated)",
    )
    nullable: Optional[str] = Field(
        default=None,
        description="Allow null values for: description, content, image (comma-separated)",
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Filter articles from this date (ISO 8601 format: YYYY-MM-DDTHH:MM:SS.sssZ)",
    )
    date_to: Optional[str] = Field(
        default=None,
        description="Filter articles until this date (ISO 8601 format: YYYY-MM-DDTHH:MM:SS.sssZ)",
    )
    sortby: Optional[Literal["publishedAt", "relevance"]] = Field(
        default="publishedAt", description="Sort by publication date or relevance"
    )
    page: Optional[int] = Field(
        default=1, description="Page number for pagination", ge=1
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'markdown' for human-readable or 'json' for programmatic use",
    )


class TopHeadlinesInput(BaseModel):
    """Input model for top headlines operations."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    category: Optional[
        Literal[
            "general",
            "world",
            "nation",
            "business",
            "technology",
            "entertainment",
            "sports",
            "science",
            "health",
        ]
    ] = Field(default="general", description="News category")
    lang: Optional[str] = Field(
        default=None,
        description=f"Language code (2 letters). Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}",
    )
    country: Optional[str] = Field(
        default=None,
        description=f"Country code (2 letters). Supported: {', '.join(SUPPORTED_COUNTRIES.keys())}",
    )
    max_articles: Optional[int] = Field(
        default=10, description="Number of articles to return (1-100)", ge=1, le=100
    )
    nullable: Optional[str] = Field(
        default=None,
        description="Allow null values for: description, content, image (comma-separated)",
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Filter articles from this date (ISO 8601 format: YYYY-MM-DDTHH:MM:SS.sssZ)",
    )
    date_to: Optional[str] = Field(
        default=None,
        description="Filter articles until this date (ISO 8601 format: YYYY-MM-DDTHH:MM:SS.sssZ)",
    )
    q: Optional[str] = Field(
        default=None, description="Additional search keywords to filter headlines"
    )
    page: Optional[int] = Field(
        default=1, description="Page number for pagination", ge=1
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'markdown' for human-readable or 'json' for programmatic use",
    )


def get_api_key() -> str:
    """Get the GNews API key from environment variables"""
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        raise ValueError(
            "GNEWS_API_KEY environment variable is required. "
            "Get your free API key from https://gnews.io/"
        )
    return api_key


def _handle_api_error(e: Exception) -> str:
    """Consistent error formatting with actionable messages."""
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code == 401:
            return "Error: Invalid API key. Please check your GNEWS_API_KEY environment variable. Get your key from https://gnews.io/"
        elif e.response.status_code == 403:
            return "Error: API access forbidden. Check your subscription plan at https://gnews.io/pricing"
        elif e.response.status_code == 429:
            return "Error: Rate limit exceeded. Please wait before making more requests or upgrade your plan."
        elif e.response.status_code == 400:
            return "Error: Invalid request parameters. Check your query syntax and parameter values."
        else:
            return f"Error: API request failed with status {e.response.status_code}"
    elif isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. The GNews API is taking too long to respond. Please try again."
    elif isinstance(e, httpx.RequestError):
        return f"Error: Network error connecting to GNews API: {str(e)}"
    return f"Error: Unexpected error occurred: {type(e).__name__} - {str(e)}"


def _format_articles_as_markdown(data: dict, query_info: str) -> str:
    """Format articles as markdown for human readability."""
    lines = [f"# GNews Results: {query_info}", ""]
    lines.append(f"**Total Articles:** {data.get('totalArticles', 0)}")
    lines.append("")

    articles = data.get("articles", [])
    for i, article in enumerate(articles, 1):
        lines.append(f"## {i}. {article.get('title', 'No title')}")
        lines.append("")
        lines.append(f"**Published:** {article.get('publishedAt', 'Unknown')}")
        lines.append(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
        lines.append("")
        if article.get("description"):
            lines.append(article["description"])
            lines.append("")
        lines.append(f"[Read more]({article.get('url', '#')})")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _format_articles_as_json(data: dict) -> str:
    """Format articles as JSON for programmatic processing."""
    return json.dumps(data, indent=2)


async def make_gnews_request(endpoint: str, params: dict) -> dict:
    """Make a request to the GNews API"""
    api_key = get_api_key()

    # Add API key to parameters
    params["apikey"] = api_key

    # Base URL for GNews API
    base_url = "https://gnews.io/api/v4"
    url = f"{base_url}/{endpoint}"

    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Making request to {endpoint} with params: {params}")
            response = await client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                logger.info(
                    f"Successfully retrieved {data.get('totalArticles', 0)} articles"
                )
                return data
            else:
                error_msg = f"GNews API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "errors" in error_data:
                        error_msg += f" - {error_data['errors']}"
                except (json.JSONDecodeError, ValueError, Exception):
                    error_msg += f" - {response.text}"

                logger.error(error_msg)
                raise httpx.HTTPStatusError(
                    error_msg, request=response.request, response=response
                )

    except httpx.RequestError as e:
        error_msg = f"Network error connecting to GNews API: {str(e)}"
        logger.error(error_msg)
        raise


@mcp.tool(
    name="gnews_search_news",
    annotations={
        "title": "Search GNews Articles",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def gnews_search_news(params: SearchNewsInput) -> str:
    """
    Search for news articles using specific keywords.

    This tool allows you to search for news articles based on keywords with various
    filtering options including language, country, date range, and sorting preferences.

    Query Syntax Examples:
    - Simple search: "Apple iPhone"
    - Exact phrase: '"Apple iPhone 15"'
    - Logical operators: "Apple AND iPhone", "Apple OR Microsoft", "Apple NOT iPhone"
    - Complex queries: "(Apple AND iPhone) OR Microsoft"

    Returns a structured response with article details including title, description,
    content, URL, image, publishedAt, and source information.

    Response can be formatted as JSON (for programmatic use) or Markdown (for human reading).
    """

    # Validate language and country
    if params.lang and params.lang not in SUPPORTED_LANGUAGES:
        return json.dumps(
            {
                "success": False,
                "error": f"Unsupported language '{params.lang}'. Supported languages: {', '.join(SUPPORTED_LANGUAGES.keys())}",
            }
        )

    if params.country and params.country not in SUPPORTED_COUNTRIES:
        return json.dumps(
            {
                "success": False,
                "error": f"Unsupported country '{params.country}'. Supported countries: {', '.join(SUPPORTED_COUNTRIES.keys())}",
            }
        )

    # Build request parameters
    api_params = {"q": params.q}

    if params.lang:
        api_params["lang"] = params.lang
    if params.country:
        api_params["country"] = params.country
    if params.max_articles:
        api_params["max"] = params.max_articles
    if params.search_in:
        api_params["in"] = params.search_in
    if params.nullable:
        api_params["nullable"] = params.nullable
    if params.date_from:
        api_params["from"] = params.date_from
    if params.date_to:
        api_params["to"] = params.date_to
    if params.sortby:
        api_params["sortby"] = params.sortby
    if params.page:
        api_params["page"] = params.page

    try:
        result = await make_gnews_request("search", api_params)

        response_data = {
            "success": True,
            "query": params.q,
            "totalArticles": result.get("totalArticles", 0),
            "articles": result.get("articles", []),
            "parameters_used": api_params,
        }

        # Format response based on requested format
        if params.response_format == ResponseFormat.MARKDOWN:
            return _format_articles_as_markdown(response_data, f"Search '{params.q}'")
        else:
            return _format_articles_as_json(response_data)

    except Exception as e:
        error_msg = _handle_api_error(e)
        error_response = {
            "success": False,
            "error": error_msg,
            "query": params.q,
            "parameters_used": api_params,
        }
        return json.dumps(error_response, indent=2)


@mcp.tool(
    name="gnews_get_top_headlines",
    annotations={
        "title": "Get GNews Top Headlines",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def gnews_get_top_headlines(params: TopHeadlinesInput) -> str:
    """
    Get current trending news articles based on Google News ranking.

    This tool retrieves the top headlines for a specific category. The articles
    are selected based on Google News ranking algorithm, providing the most
    relevant and trending news for the chosen category.

    Available categories:
    - general: General news (default)
    - world: International news
    - nation: National news
    - business: Business and finance
    - technology: Technology and innovation
    - entertainment: Entertainment and celebrity news
    - sports: Sports news
    - science: Scientific discoveries and research
    - health: Health and medical news

    Returns a structured response with trending article details.
    Response can be formatted as JSON (for programmatic use) or Markdown (for human reading).
    """

    # Validate language and country
    if params.lang and params.lang not in SUPPORTED_LANGUAGES:
        return json.dumps(
            {
                "success": False,
                "error": f"Unsupported language '{params.lang}'. Supported languages: {', '.join(SUPPORTED_LANGUAGES.keys())}",
            }
        )

    if params.country and params.country not in SUPPORTED_COUNTRIES:
        return json.dumps(
            {
                "success": False,
                "error": f"Unsupported country '{params.country}'. Supported countries: {', '.join(SUPPORTED_COUNTRIES.keys())}",
            }
        )

    # Build request parameters
    api_params = {}

    if params.category:
        api_params["category"] = params.category
    if params.lang:
        api_params["lang"] = params.lang
    if params.country:
        api_params["country"] = params.country
    if params.max_articles:
        api_params["max"] = params.max_articles
    if params.nullable:
        api_params["nullable"] = params.nullable
    if params.date_from:
        api_params["from"] = params.date_from
    if params.date_to:
        api_params["to"] = params.date_to
    if params.q:
        api_params["q"] = params.q
    if params.page:
        api_params["page"] = params.page

    try:
        logger.info(
            f"Getting top headlines for category '{params.category}' with params: {api_params}"
        )
        result = await make_gnews_request("top-headlines", api_params)

        response_data = {
            "success": True,
            "category": params.category or "general",
            "totalArticles": result.get("totalArticles", 0),
            "articles": result.get("articles", []),
            "parameters_used": api_params,
        }

        # Format response based on requested format
        if params.response_format == ResponseFormat.MARKDOWN:
            return _format_articles_as_markdown(
                response_data, f"Top Headlines: {params.category}"
            )
        else:
            return _format_articles_as_json(response_data)

    except Exception as e:
        error_msg = _handle_api_error(e)
        error_response = {
            "success": False,
            "error": error_msg,
            "category": params.category or "general",
            "parameters_used": api_params,
        }
        return json.dumps(error_response, indent=2)
