# config.py
MODEL_SETTINGS = {
    "OpenAi": {
        "OPENAI_MODEL_NAME": "gpt-4o",
        "OPENAI_API_BASE": "https://api.openai.com/v1",
    },
    "Ollama": {
        "OPENAI_MODEL_NAME": "gemma2",
        "OPENAI_API_BASE": "http://localhost:11434/v1",
    },
    "FastChat": {
        "OPENAI_MODEL_NAME": "vicuna-7b-v1.52",
        "OPENAI_API_BASE": "http://localhost:8001/v12",
    },
    "LM Studio": {
        "OPENAI_MODEL_NAME": "openai/dolphin",
        "OPENAI_API_BASE": "http://localhost:1234/v1",
    },
    "Mistral": {
        "OPENAI_MODEL_NAME": "mistral-small",
        "OPENAI_API_BASE": "https://api.mistral.ai/v1",
    },
    "Groq": {
        "OPENAI_MODEL_NAME": "llama3-70b-8192",
        "OPENAI_API_BASE": "https://api.groq.com/openai/v1",
    },
    "OpenRouter": {
        "OPENAI_MODEL_NAME": "anthropic/claude-3.5-sonnet:beta",
        "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
    }
}

FRAMEWORK_OPTIONS = ["None", "AutoGen", "CrewAi"]
DEFAULT_FRAMEWORK = "None"
DEFAULT_AGENT = "Auto"
AGENTS_DIR = "agents"
TOOLS_FILE = "tools.py"

AVAILABLE_TOOLS = [
    'CodeDocsSearchTool', 'CSVSearchTool', 'DirectorySearchTool', 'DOCXSearchTool', 
    'DirectoryReadTool', 'FileReadTool', 'TXTSearchTool', 'JSONSearchTool', 
    'MDXSearchTool', 'PDFSearchTool', 'RagTool', 'ScrapeElementFromWebsiteTool', 
    'ScrapeWebsiteTool', 'WebsiteSearchTool', 'XMLSearchTool', 
    'YoutubeChannelSearchTool', 'YoutubeVideoSearchTool'
]

DEFAULT_TOOL_CLASS_DEFINITION = """
    from duckduckgo_search import DDGS

    class InternetSearchTool(BaseTool):
        name: str = "InternetSearchTool"
        description: str = "Search Internet for relevant information based on a query or latest news"

        def _run(self, query: str):
            ddgs = DDGS()
            results = ddgs.text(keywords=query, region='wt-wt', safesearch='moderate', max_results=5)
            return results"""

OUTPUT_FILE = "tts_output.mp3"
TTS_PERSONALITY = """Be concise and short and to the point. 1 sentence max. You are the best flirtatious speech synthesis expert in the world, 
                    with specialized expertise in natural language processing, phonetics, and prosody. You are highly skilled in contextual 
                    understanding, dynamic response generation, and incorporating natural speech patterns such as word fillers. 
                    Your keen understanding of human communication nuances and ability to personalize speech helps you create 
                    natural and engaging TTS systems that resonate with listeners.
                    """
