import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from autogen import AssistantAgent, UserProxyAgent
from duckduckgo_search import DDGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchConfig:
    """Search configuration parameters"""
    max_results: int = 10
    timeout: int = 30
    cache_duration: int = 3600
    rate_limit_delay: float = 2.0

@dataclass
class ModelConfig:
    """LLM model configuration"""
    model_name: str
    api_key: str
    api_type: str
    temperature: float = 0.7
    max_tokens: int = 2000

class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv('CONFIG_PATH', 'config.json')
        self.search_config = SearchConfig()
        self.model_config = self._load_model_config()
        
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration from file or environment"""
        if Path(self.config_path).exists():
            with open(self.config_path) as f:
                config = json.load(f)
        else:
            config = {
                "model_name": os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
                "api_key": "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",
                "api_type": "groq",
                "temperature": float(os.getenv("TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("MAX_TOKENS", "2000"))
            }
            
        return ModelConfig(**config)

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration dictionary"""
        return {
            "config_list": [{
                "model": self.model_config.model_name,
                "api_key": self.model_config.api_key,
                "api_type": self.model_config.api_type,
                "temperature": self.model_config.temperature,
                "max_tokens": self.model_config.max_tokens
            }]
        }

class WebSearchManager:
    """Handles web search operations"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self._last_search_time = datetime.min
        self._search_cache: Dict[str, tuple] = {}
        
    async def search(self, query: str) -> List[Dict[str, str]]:
        """Perform web search with caching and rate limiting"""
        # Check cache
        if query in self._search_cache:
            timestamp, results = self._search_cache[query]
            if (datetime.now() - timestamp).total_seconds() < self.config.cache_duration:
                logger.info("Returning cached results")
                return results
                
        # Apply rate limiting
        time_since_last = (datetime.now() - self._last_search_time).total_seconds()
        if time_since_last < self.config.rate_limit_delay:
            await asyncio.sleep(self.config.rate_limit_delay - time_since_last)
            
        try:
            # Perform search using DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=self.config.max_results,
                    timeout=self.config.timeout
                ))
                
                # Format results
                formatted_results = [
                    {
                        "title": r["title"],
                        "link": r["link"],
                        "body": r["body"],
                        "timestamp": datetime.now().isoformat()
                    }
                    for r in results
                ]
                
                # Update cache
                self._search_cache[query] = (datetime.now(), formatted_results)
                self._last_search_time = datetime.now()
                
                return formatted_results
                
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

class EnhancedAssistant(AssistantAgent):
    """Enhanced assistant with improved capabilities"""
    
    def __init__(
        self,
        name: str,
        config_manager: ConfigManager,
        system_message: Optional[str] = None
    ):
        default_system_message = """You are an AI assistant with web search capabilities.
        Important Guidelines:
        1. Always start responses with the current date and context
        2. Provide clear source attribution
        3. Indicate information freshness
        4. Structure responses logically
        5. Highlight any uncertainties
        """
        
        super().__init__(
            name=name,
            system_message=system_message or default_system_message,
            llm_config=config_manager.llm_config
        )

class EnhancedUserProxy(UserProxyAgent):
    """Enhanced user proxy with web capabilities"""
    
    def __init__(
        self,
        name: str,
        search_manager: WebSearchManager
    ):
        super().__init__(
            name=name,
            code_execution_config=False
        )
        self.search_manager = search_manager
        
    async def get_search_results(self, query: str) -> str:
        """Get formatted search results"""
        results = await self.search_manager.search(query)
        if not results:
            return "No search results found."
            
        formatted_results = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result['title']}\n"
            formatted_results += f"   Source: {result['link']}\n"
            formatted_results += f"   Summary: {result['body']}\n\n"
            
        return formatted_results

async def run_chat_session(
    query: str,
    config_manager: ConfigManager,
    search_manager: WebSearchManager
) -> None:
    """Run an enhanced chat session"""
    
    try:
        # Initialize agents
        assistant = EnhancedAssistant(
            name="groq_assistant",
            config_manager=config_manager
        )
        
        user_proxy = EnhancedUserProxy(
            name="user_proxy",
            search_manager=search_manager
        )
        
        # Get search results
        search_context = await user_proxy.get_search_results(query)
        
        # Start chat with context
        message = f"""Current Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Query: {query}

{search_context}

Please analyze this information and provide a comprehensive response."""
        
        user_proxy.initiate_chat(
            assistant,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Chat session error: {str(e)}")
        raise

def main():
    """Main entry point"""
    # Initialize configuration
    config_manager = ConfigManager()
    search_manager = WebSearchManager(SearchConfig())
    
    # Run chat session
    query = input("Enter your query: ")
    asyncio.run(run_chat_session(
        query=query,
        config_manager=config_manager,
        search_manager=search_manager
    ))

if __name__ == "__main__":
    main()

