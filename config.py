# config.py

MODEL_SETTINGS = {
    "OpenAi": {
        "OPENAI_MODEL_NAME": "gpt-4o",
        "OPENAI_API_BASE": "https://api.openai.com/v1",
    },
    "Ollama Mistral": {
        "OPENAI_MODEL_NAME": "mistral",
        "OPENAI_API_BASE": "http://localhost:11434/v1",
    },
    "FastChat": {
        "OPENAI_MODEL_NAME": "oh-2.5m7b-q51",
        "OPENAI_API_BASE": "http://localhost:8001/v1",
    },
    "LM Studio": {
        "OPENAI_MODEL_NAME": "NA",
        "OPENAI_API_BASE": "http://localhost:8000/v1",
    },
    "Mistral": {
        "OPENAI_MODEL_NAME": "mistral-small",
        "OPENAI_API_BASE": "https://api.mistral.ai/v1",
    },
    "Groq": {
        "OPENAI_MODEL_NAME": "llama3-70b-8192",
        "OPENAI_API_BASE": "https://api.groq.com/openai/v1",
    }
}

FRAMEWORK_OPTIONS = ["None", "AutoGen", "CrewAi", "Battle"]
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
