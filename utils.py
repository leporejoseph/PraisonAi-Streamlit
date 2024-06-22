# utils.py

import os
import yaml
import json
import streamlit as st
from openai import OpenAI
from config import AGENTS_DIR, TOOLS_FILE
import re

CONVERSATION_HISTORY_FILE = 'conversation_history.json'

def initialize_env():
    default_values = {
        "OPENAI_MODEL_NAME": "Enter Model Name Here",
        "OPENAI_API_BASE": "Enter API Base Here",
        "OPENAI_API_KEY": "Enter API Key Here",
        "OPENAI_LLM_API_KEY": "Enter API Key Here",
        "OLLAMA_MISTRAL_API_KEY": "NA",
        "FASTCHAT_API_KEY": "NA",
        "LM_STUDIO_API_KEY": "NA",
        "MISTRAL_API_API_KEY": "Enter API Key Here",
        "GROQ_API_KEY": "Enter API Key Here"
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

    if model_name.lower() == "openai":
        env_vars["OPENAI_LLM_API_KEY"] = api_key
    else:
        env_vars[f"{model_name.upper()}_API_KEY"] = api_key

    with open(env_path, 'w') as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")

    os.environ.update(env_vars)

    st.session_state.client = OpenAI(api_key=env_vars["OPENAI_LLM_API_KEY"] if model_name.lower() == "openai" else api_key, base_url=api_base)

def get_api_key(model_name):
    if model_name.lower() == "openai":
        return os.getenv("OPENAI_LLM_API_KEY", "Enter API Key Here")
    return os.getenv(f"{model_name.upper()}_API_KEY", "NA" if model_name in ["ollama_mistral", "fastchat", "lm_studio"] else "Enter API Key Here")

def get_agents_list():
    agents_dir = 'agents'
    agents_files = ["Auto Generate New Agents"]

    if os.path.exists(agents_dir):
        agents_files.extend(f for f in os.listdir(agents_dir) if f.endswith('.yaml'))

    return agents_files

def rename_and_move_yaml():
    agents_dir = 'agents'
    if not os.path.exists(agents_dir):
        os.makedirs(agents_dir)
    
    existing_agents = [f for f in os.listdir(agents_dir) if f.startswith('agent_') and f.endswith('.yaml')]
    new_agent_number = len(existing_agents) + 1
    new_agent_filename = f'agent_{new_agent_number}.yaml'

    if os.path.exists('test.yaml'):
        os.rename('test.yaml', os.path.join(agents_dir, new_agent_filename))
        return new_agent_filename
    else:
        raise FileNotFoundError("The file 'test.yaml' does not exist.")

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
    config['llm_model'] = llm_model
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

def save_tool_to_file(name, class_definition):
    if not os.path.exists(TOOLS_FILE):
        with open(TOOLS_FILE, 'w') as file:
            file.write('#tools.py\n\n\n')

    with open(TOOLS_FILE, 'r') as file:
        content = file.readlines()

    # Extract existing imports
    existing_imports = []
    for line in content:
        if line.strip().startswith('from') or line.strip().startswith('import'):
            existing_imports.append(line.strip())

    # Extract new imports from class_definition
    new_imports = []
    class_lines = []
    for line in class_definition.split('\n'):
        if line.strip().startswith('from') or line.strip().startswith('import'):
            new_imports.append(line.strip())
        else:
            class_lines.append(line)

    # Combine and deduplicate imports
    all_imports = sorted(set(existing_imports + new_imports))

    # Prepare new content
    new_content = ['#tools.py\n\n\n']
    new_content.extend(f'{imp}\n' for imp in all_imports)
    if all_imports:
        new_content.append('\n')

    # Add existing tools
    tool_started = False
    for line in content:
        if line.strip().startswith('# ') and not tool_started:
            tool_started = True
            new_content.append(line)
        elif line.strip().startswith('# ') and tool_started:
            new_content.append(line)
            tool_started = False
        elif tool_started and not (line.strip().startswith('from') or line.strip().startswith('import')):
            new_content.append(line)

    # Add the new tool
    new_content.append(f'\n# {name}\n')
    new_content.extend(f'{line}\n' for line in class_lines)
    new_content.append(f'# {name}\n')

    # Write the updated content back to the file
    with open(TOOLS_FILE, 'w') as file:
        file.writelines(new_content)

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

