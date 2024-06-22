# utils.py

import os
import yaml
import json
import streamlit as st
from openai import OpenAI
import anthropic
from config import AGENTS_DIR, TOOLS_FILE, MODEL_SETTINGS
import re

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
    from config import MODEL_SETTINGS
    settings = MODEL_SETTINGS[model_name]

    env_path = '.env'
    env_vars = {}

    if os.path.exists(env_path):
        with open(env_path, 'r') as file:
            env_vars = dict(line.strip().split('=', 1) for line in file if '=' in line)

    env_vars.update({
        "OPENAI_MODEL_NAME": settings["OPENAI_MODEL_NAME"],
        "OPENAI_API_BASE": api_base,
        "OPENAI_API_KEY": api_key
    })

    model_key = model_name.upper().replace(' ', '_') + "_API_KEY"
    if model_name.lower() == "openai":
        env_vars["OPENAI_LLM_API_KEY"] = api_key
    else:
        env_vars[model_key] = api_key

    with open(env_path, 'w') as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")

    os.environ.update(env_vars)

    if model_name.lower() == "anthropic":
        st.session_state.client = anthropic.Anthropic(api_key=env_vars["ANTHROPIC_API_KEY"])
    else:
        st.session_state.client = OpenAI(api_key=env_vars["OPENAI_LLM_API_KEY"] if model_name.lower() == "openai" else api_key, base_url=api_base)

def get_api_key(model_name):
    model_key = model_name.upper().replace(' ', '_') + "_API_KEY"
    if model_name.lower() == "openai":
        return os.getenv("OPENAI_LLM_API_KEY", "Enter API Key Here")
    return os.getenv(model_key, "NA" if model_name in ["ollama_mistral", "fastchat", "lm_studio"] else "Enter API Key Here")

def get_agents_list():
    agents_dir = 'agents'
    agents_files = ["Auto Generate New Agents"]

    if os.path.exists(agents_dir):
        agents_files.extend(f for f in os.listdir(agents_dir) if f.endswith('.yaml'))

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

def save_conversation_history(messages):
    with open(CONVERSATION_HISTORY_FILE, 'w') as file:
        json.dump(messages, file, indent=4)

def clear_conversation_history():
    with open(CONVERSATION_HISTORY_FILE, 'w') as file:
        json.dump([], file, indent=4)
    st.session_state.messages = []

def load_selected_llm_provider():
    if os.path.exists('config.json'):
        with open('config.json', 'r') as file:
            config = json.load(file)
            return config.get('llm_model', 'OpenAi')
    return 'OpenAi'

def save_selected_llm_provider(llm_model):
    config = {}
    if os.path.exists('config.json'):
        with open('config.json', 'r') as file:
            config = json.load(file)
    
    config['llm_model'] = 'Local' if st.session_state.llm_model == 'Local' else llm_model
    
    with open('config.json', 'w') as file:
        json.dump(config, file, indent=4)

def initialize_session_state():
    if 'llm_model' not in st.session_state:
        st.session_state['llm_model'] = load_selected_llm_provider()
    if 'client' not in st.session_state:
        if st.session_state.llm_model.lower() == "anthropic":
            st.session_state['client'] = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
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
        st.session_state['local_model'] = "LM Studio"

def load_tools_from_file(tools_file):
    if not os.path.exists(tools_file):
        with open(tools_file, 'w') as file:
            file.write('')  # Create the file if it doesn't exist

    tools = {}
    with open(tools_file, 'r') as file:
        content = file.read()
        
    # Split the content by tool definitions
    tool_definitions = content.split('# ')[1:]  # Skip the first split which is likely imports
    
    for definition in tool_definitions:
        lines = definition.strip().split('\n')
        tool_name = lines[0].strip()  # The tool name is the first line after '#'
        class_def = '\n'.join(lines[1:]).strip()  # The rest is the class definition
        
        if 'class' in class_def:
            class_name = class_def.split('class ')[1].split('(')[0].strip()
            tools[class_name] = f"# {tool_name}\n{class_def}"

    return tools

def edit_tool_in_file(tool_name, new_tool_code):
    with open(TOOLS_FILE, 'r') as file:
        content = file.read()

    # Define the regex pattern to find the specific tool block
    pattern = re.compile(rf"(# {tool_name}\n).*?(# {tool_name}\n)", re.DOTALL)

    # Replace the old tool code with the new tool code
    new_content = re.sub(pattern, new_tool_code, content)

    with open(TOOLS_FILE, 'w') as file:
        file.write(new_content)

def load_tool_class_definition(tool_name):
    with open(TOOLS_FILE, 'r') as file:
        content = file.read()

    tool_definitions = content.split(f'# {tool_name}')
    if len(tool_definitions) >= 3:
        return f"# {tool_name}{tool_definitions[1]}# {tool_name}"
    return ""  # Return empty string if tool not found

def delete_tool_from_file(tool_name):
    with open(TOOLS_FILE, 'r') as file:
        content = file.read()

    # Define the regex pattern to find the specific tool block including comments
    pattern = re.compile(rf"# {tool_name}\n.*?# {tool_name}\n", re.DOTALL)

    # Remove the tool block
    new_content = re.sub(pattern, '', content)

    with open(TOOLS_FILE, 'w') as file:
        file.write(new_content.strip() + '\n')  # Ensure file ends with a newline

def is_local_model(model_name):
    return 'localhost' in MODEL_SETTINGS[model_name]['OPENAI_API_BASE']
