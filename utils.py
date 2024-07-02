# utils.py

import os
import yaml
import json
import streamlit as st
from openai import OpenAI
from config import AGENTS_DIR, TOOLS_FILE, MODEL_SETTINGS
import re
from datetime import datetime
from groq import Groq
from interpreter import OpenInterpreter
from PIL import Image
from io import BytesIO
import base64
from random import randint

import asyncio
import edge_tts

CONVERSATION_HISTORY_FILE = 'conversation_history.json'

def initialize_env():
    default_values = {
        "OPENAI_MODEL_NAME": "Enter Model Name Here",
        "OPENAI_API_BASE": "Enter API Base Here",
        "OPENAI_API_KEY": "Enter API Key Here",
        "OPENAI_LLM_API_KEY": "Enter API Key Here",
        "OLLAMA_MISTRAL_API_KEY": "OPTIONAL",
        "FASTCHAT_API_KEY": "OPTIONAL",
        "LM_STUDIO_API_KEY": "OPTIONAL",
        "MISTRAL_API_API_KEY": "Enter API Key Here",
        "GROQ_API_KEY": "Enter API Key Here",
        "ANTHROPIC_API_KEY": "Enter API Key Here"
    }

    env_path = '.env'
    env_vars = {}

    if os.path.exists(env_path):
        with open(env_path, 'r') as file:
            env_vars = dict(line.strip().split('=', 1) for line in file if '=' in line)

    env_vars.update({key: env_vars.get(key, value) for key, value in default_values.items()})

    with open(env_path, 'w') as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")

    os.environ.update(env_vars)

def update_env(model_name, api_base, api_key):
    env_path = '.env'
    env_vars = {}

    if os.path.exists(env_path):
        with open(env_path, 'r') as file:
            env_vars = dict(line.strip().split('=', 1) for line in file if '=' in line)

    # Determine the appropriate API key variable name
    llm_key_prefix = st.session_state.get("local_model", "") if st.session_state.llm_model == "Local" else st.session_state.llm_model
    api_key_var_name = f"{llm_key_prefix.upper().replace(' ', '_')}_API_KEY"
    env_vars.update({
        "OPENAI_MODEL_NAME": model_name,
        "OPENAI_API_BASE": api_base,
        "OPENAI_API_KEY": api_key,
        api_key_var_name: api_key
    })

    with open(env_path, 'w') as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")

    os.environ.update(env_vars)

    # Initialize the client based on the model
    st.session_state.client = OpenAI(api_key=env_vars.get(api_key_var_name), base_url=api_base)

def update_model_settings(llm_provider, model_name, api_base):
    config_path = 'config.py'

    with open(config_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    in_model_section = False

    for line in lines:
        if f'"{llm_provider}":' in line:
            in_model_section = True
        if in_model_section and 'OPENAI_MODEL_NAME' in line:
            new_lines.append(f'        "OPENAI_MODEL_NAME": "{model_name}",\n')
            continue
        if in_model_section and 'OPENAI_API_BASE' in line:
            new_lines.append(f'        "OPENAI_API_BASE": "{api_base}",\n')
            in_model_section = False
            continue
        new_lines.append(line)

    with open(config_path, 'w') as file:
        file.writelines(new_lines)

def save_config(key, value):
    config = {}
    config_path = 'config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
    config[key] = value
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

def save_tts_settings(enable_tts, enhance_tts, tts_personality):
    save_config("enable_tts", enable_tts)
    save_config("enhance_tts", enhance_tts)
    save_config("tts_personality", tts_personality)

def load_tts_settings():
    config_path = 'config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config.get("enable_tts", True), config.get("enhance_tts", False), config.get("tts_personality", "en-GB-SoniaNeural")
    return True, False, "en-GB-SoniaNeural"

def get_api_key(model_name):
    model_key = model_name.upper().replace(' ', '_') + "_API_KEY"
    if model_name.lower() == "openai":
        return os.getenv("OPENAI_LLM_API_KEY", "Enter API Key Here")
    return os.getenv(model_key, "NA" if model_name in ["ollama_mistral", "fastchat", "lm_studio"] else "Enter API Key Here")

def get_agents_list():
    agents_files = []
    if os.path.exists(AGENTS_DIR):
        agents_files.extend(f for f in os.listdir(AGENTS_DIR) if f.endswith('.yaml'))
    return agents_files

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False, indent=4)

def load_conversation_history():
    if os.path.exists(CONVERSATION_HISTORY_FILE):
        with open(CONVERSATION_HISTORY_FILE, 'r') as file:
            return json.load(file)
    return []

@st.experimental_fragment
def save_conversation_history(messages):
    with open(CONVERSATION_HISTORY_FILE, 'w') as file:
        json.dump(messages, file, indent=4)

@st.experimental_fragment
def clear_conversation_history():
    with open(CONVERSATION_HISTORY_FILE, 'w') as file:
        json.dump([], file, indent=4)
    st.session_state.messages = []
    st.rerun()

def load_selected_llm_provider():
    if os.path.exists('config.json'):
        with open('config.json', 'r') as file:
            config = json.load(file)
            return config.get('llm_provider', 'OpenAi')
    return 'OpenAi'

def save_selected_llm_provider(llm_provider):
    config = {}
    if os.path.exists('config.json'):
        with open('config.json', 'r') as file:
            config = json.load(file)
    config['llm_provider'] = 'Local' if st.session_state.llm_model == 'Local' else llm_provider
    with open('config.json', 'w') as file:
        json.dump(config, file, indent=4)

def initialize_session_state():
    if 'llm_model' not in st.session_state:
        st.session_state['llm_model'] = load_selected_llm_provider()
    if 'client' not in st.session_state:
        st.session_state['client'] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
    if 'api_key' not in st.session_state:
        st.session_state.api_key = get_api_key(st.session_state.llm_model)
    if 'show_edit_container' not in st.session_state:
        st.session_state['show_edit_container'] = False
    if 'tools' not in st.session_state:
        st.session_state['tools'] = load_tools_from_file(TOOLS_FILE)
    if 'messages' not in st.session_state:
        st.session_state['messages'] = load_conversation_history()
    if 'local_model' not in st.session_state:
        st.session_state.local_model = "LM Studio"
    if 'api_base' not in st.session_state:
        st.session_state.api_base = os.getenv("OPENAI_API_BASE")
    if 'model_name' not in st.session_state:
        st.session_state.model_name = os.getenv("OPENAI_MODEL_NAME")
    if 'enable_tts' not in st.session_state:
        enable_tts, enhance_tts, tts_personality = load_tts_settings()
        st.session_state.enable_tts = enable_tts
        st.session_state.enhance_tts = enhance_tts
        st.session_state.tts_personality = tts_personality
    if 'widget_key' not in st.session_state:
        st.session_state.widget_key = str(randint(1000, 100000000))
    if 'config_list' not in st.session_state:
        st.session_state.config_list = [{
            'model': st.session_state.model_name,
            'base_url': st.session_state.api_base,
            'api_key': st.session_state.api_key,
            'api_type': st.session_state.llm_model.lower()
        }]
    if 'open_interpreter' not in st.session_state:
        st.session_state.open_interpreter = OpenInterpreter()
        st.session_state.open_interpreter.llm.model = st.session_state.model_name
        st.session_state.open_interpreter.llm.api_base = st.session_state.api_base
        st.session_state.open_interpreter.llm.api_key = st.session_state.api_key

def load_tools_from_file(tools_file):
    if not os.path.exists(tools_file):
        with open(tools_file, 'w') as file:
            file.write('')  # Create the file if it doesn't exist

    tools = {}
    with open(tools_file, 'r') as file:
        content = file.read()

    # Use regular expressions to find all class definitions in the file
    import re
    pattern = re.compile(r'class\s+(\w+)\s*\((BaseTool|.*)\):')
    matches = pattern.findall(content)
    
    for match in matches:
        class_name = match[0]
        tools[class_name] = content.split(f'class {class_name}(')[1].split('\n')[0]
    
    return tools

def is_local_model(model_name):
    return 'localhost' in MODEL_SETTINGS[model_name]['OPENAI_API_BASE']

def transcribe_audio(uploaded_file):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    transcription = client.audio.transcriptions.create(
        file=(uploaded_file.name, uploaded_file.getvalue()),
        model="whisper-large-v3",
        response_format="verbose_json"
    )
    return transcription

def save_transcription_to_file(transcription, original_filename):
    directory = "documents"
    if not os.path.exists(directory):
        os.makedirs(directory)

    timestamp = datetime.now().strftime("%m_%d_%Y_%H%M%S")
    filename_without_ext = os.path.splitext(original_filename)[0]
    filepath = os.path.join(directory, f"{filename_without_ext}_{timestamp}.json")

    filtered_transcription = {
        "text": transcription.text,
        "segments": [
            {
                "start": segment['start'],
                "end": segment['end'],
                "text": segment['text']
            }
            for segment in transcription.segments
        ]
    }

    with open(filepath, 'w') as f:
        json.dump(filtered_transcription, f, indent=4)

def documents_exist(directory="documents"):
    return os.path.exists(directory) and len(os.listdir(directory)) > 0

def list_documents(directory="documents"):
    if documents_exist(directory):
        return os.listdir(directory)
    return []

def load_document_content(filepath):
    with open(filepath, 'r') as file:
        content = json.load(file)
    return content.get("text", "")

def update_document_content(filepath, new_content):
    with open(filepath, 'r') as file:
        content = json.load(file)
    content["text"] = new_content
    with open(filepath, 'w') as file:
        json.dump(content, file, indent=4)


def format_response(chunk, full_response):
    if chunk['type'] == "message":
        full_response += chunk.get("content", "")
        if chunk.get('end', False):
            full_response += "\n"

    if chunk['type'] == "code":
        if chunk.get('start', False):
            full_response += "```python\n"
        full_response += chunk.get('content', '')
        if chunk.get('end', False):
            full_response += "\n```\n"

    if chunk['type'] == "confirmation":
        if chunk.get('start', False):
            full_response += "```python\n"
        full_response += chunk.get('content', {}).get('code', '')
        if chunk.get('end', False):
            full_response += "```\n"

    if chunk['type'] == "console":
        if chunk.get('start', False):
            full_response += "```python\n"
        if chunk.get('format', '') == "active_line":
            console_content = chunk.get('content', '')
            if console_content is None:
               full_response += "No output available on console."
        if chunk.get('format', '') == "output":
            console_content = chunk.get('content', '')
            full_response += console_content
        if chunk.get('end', False):
            full_response += "\n```\n"

    if chunk['type'] == "image":
        if chunk.get('start', False) or chunk.get('end', False):
            full_response += "\n"
        else:
            image_format = chunk.get('format', '')
            if image_format == 'base64.png':
                image_content = chunk.get('content', '')
                if image_content:
                    image = Image.open(BytesIO(base64.b64decode(image_content)))
                    new_image = Image.new("RGB", image.size, "white")
                    new_image.paste(image, mask=image.split()[3])
                    buffered = BytesIO()
                    new_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    full_response += f"![Image](data:image/png;base64,{img_str})\n"

    return full_response

async def synthesize_text_to_speech(text: str, voice: str, output_file: str) -> None:
    """Synthesize text to speech and save to an output file."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

async def get_text_to_speech_voices() -> list:
    return await edge_tts.list_voices()

def move_and_rename_file(filename, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Get the list of existing files in the agents folder
    existing_files = get_agents_list()
    new_file_index = len(existing_files) + 1
    new_filename = f"Agents_{new_file_index}.yaml"

    os.rename(filename, os.path.join(target_folder, new_filename))