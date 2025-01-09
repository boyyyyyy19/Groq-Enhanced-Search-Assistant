# Groq-Enhanced-Search-Assistant

A sophisticated AI assistant that integrates web search functionality, enabling it to provide real-time, detailed, and contextually accurate responses based on user queries. This system utilizes advanced language models along with a caching and rate-limiting mechanism to deliver optimal search results.

Features
Real-time Web Search: Uses DuckDuckGo to fetch relevant search results, ensuring privacy and accuracy.
Search Caching: Results are cached for a specified duration to minimize repeated queries and reduce response times.
Rate Limiting: Enforces a rate limit between searches to avoid overwhelming external services.
Enhanced Assistant: The assistant analyzes search results and integrates them into the conversation with high-level contextual understanding.
Custom Configuration: Easily configurable for different models and search parameters.
Technologies
Python 3.x: The backend language for this project.
aiohttp: For asynchronous web requests.
DuckDuckGo Search API: For web search results.
autogen: To integrate custom assistant agents.
JSON: For configuration management.
Datetime & Pathlib: For managing timestamps and file paths.
Requirements
To run this project, you will need Python 3.7 or higher.

Install Dependencies
You can install the required dependencies using pip:

The assistant will fetch search results based on your query, integrate them into its response, and present a comprehensive and well-structured answer.

How it Works
Search Manager: Handles the search operations using DuckDuckGo, caching results, and enforcing rate limits.
Enhanced Assistant: A custom assistant agent that uses the results from the search manager to build responses.
User Proxy: Simulates a user querying the system and retrieves relevant search results.
Config Manager: Manages configuration and model setup, ensuring that parameters are loaded correctly.
Contributing
Feel free to fork this project and submit pull requests. If you find bugs or have suggestions for improvements, please open an issue.

