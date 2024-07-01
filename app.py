# app.py

from praisonai import PraisonAI
from praisonai.agents_generator import AgentsGenerator
from praisonai.auto import AutoGenerator
import streamlit as st
import os
from utils import *
from config import *
import asyncio

st.set_page_config(layout="wide", page_title="PraisonAI Chatbot", page_icon=":robot_face:")

initialize_env()
initialize_session_state()

def update_model():
    llm_provider = st.session_state.llm_model if st.session_state.llm_model != "Local" else st.session_state.local_model
    api_key = get_api_key(llm_provider)
    api_base = MODEL_SETTINGS[llm_provider].get("OPENAI_API_BASE", "")
    model_name = MODEL_SETTINGS[llm_provider]["OPENAI_MODEL_NAME"]
    update_env(model_name, api_base, api_key)
    save_selected_llm_provider(llm_provider)

    st.session_state.api_base = api_base
    st.session_state.model_name = model_name
    st.session_state.api_key = api_key 
    st.session_state.config_list[0]['api_type'] = st.session_state.llm_model.lower()

def update_model_settings_field():
    llm_provider = st.session_state.llm_model if st.session_state.llm_model != "Local" else st.session_state.local_model
    update_env(st.session_state.model_name, st.session_state.api_base, st.session_state.api_key)
    update_model_settings(llm_provider, st.session_state.model_name, st.session_state.api_base)

def generate_response(framework_name, prompt, agent):
    config_list = st.session_state.config_list
    
    if agent == "Auto Generate New Agents":
        return generate_auto_response(framework_name, prompt, config_list)
    else:
        return generate_praisonai_response(framework_name, agent, prompt)

def generate_open_interpreter_response(prompt):
    oi_agent = st.session_state.open_interpreter
    oi_agent.offline = st.session_state.llm_model not in ["OpenAi", "Anthropic", "Groq"]
    oi_agent.llm.model = st.session_state.model_name
    oi_agent.llm.api_base = st.session_state.api_base
    oi_agent.llm.api_key = st.session_state.api_key
    oi_agent.llm.temperature = 0
    oi_agent.llm.context_window = 8192
    oi_agent.llm.max_tokens = 4000
    oi_agent.auto_run = True
    oi_agent.anonymized_telemetry = False

    response = "### Open Interpreter Response ###\n"
    message_placeholder = st.empty()

    for chunk in oi_agent.chat([{"role": "user", "type": "message", "content": prompt}], display=False, stream=True):
        response = format_response(chunk, response)
        message_placeholder.markdown(response + "▌")
        message_placeholder.markdown(response)

    return response

def generate_auto_response(framework_name, prompt, config_list):
    st.write(config_list)
    agent_file_path = f"{AGENTS_DIR}/Agents_{len(os.listdir(AGENTS_DIR)) + 1}.yaml"
    generator = AutoGenerator(topic=prompt, agent_file=agent_file_path, framework=framework_name.lower())
    agent_file = generator.generate()
    agents_generator = AgentsGenerator(agent_file, framework_name.lower(), config_list)
    response = agents_generator.generate_crew_and_kickoff()

    response_header = f"### {framework_name} Response ###\n"
    response = response_header + response

    st.markdown(response)
    return response

def generate_praisonai_response(framework_name, agent, prompt):
    agent_file_path = f"{AGENTS_DIR}/{agent}"
    config_list = st.session_state.config_list
            
    agents_generator = AgentsGenerator(agent_file_path, framework_name.lower(), config_list)
    response = agents_generator.generate_crew_and_kickoff()

    response_header = f"### {framework_name} Response ###\n"
    response = response_header + response

    st.markdown(response)
    return response

def fix_messages(messages):
    fixed_messages = []
    last_role = "assistant"
    
    for message in messages:
        if last_role == "assistant" and message["role"] != "user":
            fixed_messages.append({"role": "user", "content": "Run Agentic Job"})
        elif last_role == "user" and message["role"] != "assistant":
            fixed_messages.append({"role": "assistant", "content": "Try Again"})
        
        fixed_messages.append(message)
        last_role = message["role"]
    
    # Ensure it starts with "user"
    if fixed_messages and fixed_messages[0]["role"] != "user":
        fixed_messages.insert(0, {"role": "user", "content": "Run Agentic Job"})
    
    return fixed_messages

@st.experimental_fragment
def generate_llm_response():
    if st.session_state.llm_model.lower() == "anthropic":
        formatted_messages = fix_messages(st.session_state.messages)

        stream_response = st.empty()
        with st.session_state.client.messages.stream(
            max_tokens=1024,
            messages=[{"role": m["role"], "content": m["content"]} for m in formatted_messages],
            model=st.session_state["model_name"],
        ) as stream:
            response = ""
            for text in stream.text_stream:
                response += text
                stream_response.markdown(response)
            return response

    else:
        stream = st.session_state.client.chat.completions.create(
            model=st.session_state["model_name"],
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        )
        response = st.write_stream(stream)

    return response

@st.experimental_dialog("Modify Agents", width="large")
def edit_agent_dialog():
    with st.form("edit_agent_form", border=False):
        agent = st.session_state.get("agent", "")
        yaml_data = load_yaml(f"{AGENTS_DIR}/{agent}")
        updated_roles = {role_name: get_updated_role(role_name, role_data, i) for i, (role_name, role_data) in enumerate(yaml_data.get("roles", {}).items())}
        
        if st.form_submit_button("Update"):
            yaml_data["topic"] = st.text_area("Topic", value=yaml_data.get("topic", ""))
            yaml_data["roles"] = updated_roles
            save_yaml(yaml_data, f"{AGENTS_DIR}/{agent}")
            st.toast(f":robot_face: {agent} has been updated.")
            st.rerun()

def get_updated_role(role_name, role_data, container_index):
    with st.container(border=True):
        st.header("Agent")
        tools = [tool for tool in role_data.get("tools", []) if tool]
        if isinstance(tools, str):
            tools = [] if tools == '' else [tools]

        role_info = {
            "role": st.text_input("Agent Role", value=role_data.get("role", ""), key=f"role_{role_name}"),
            "tools": st.multiselect("Tools", options=get_all_tools(), default=tools, key=f"tools_{role_name}"),
            "backstory": st.text_area("Backstory", value=role_data.get("backstory", ""), key=f"backstory_{role_name}"),
            "goal": st.text_area("Goal", value=role_data.get("goal", ""), key=f"goal_{role_name}"),
            "tasks": {task_name: get_updated_task(role_name, task_name, task_data) for task_name, task_data in role_data.get("tasks", {}).items()}
        }
        
        return role_info

def get_updated_task(role_name, task_name, task_data):
    with st.container(border=False):
        st.header("Task")
        return {
            "description": st.text_area(f"Description: {task_name.replace('_', ' ').title()}", value=task_data.get("description", ""), key=f"description_{role_name}_{task_name}"),
            "expected_output": st.text_area(f"Expected Output: {task_name.replace('_', ' ').title()}", value=task_data.get("expected_output", ""), key=f"expected_output_{role_name}_{task_name}")
        }

def get_all_tools():
    return sorted(set(AVAILABLE_TOOLS + list(load_tools_from_file(TOOLS_FILE).keys())))

@st.experimental_dialog("Modify Tools", width="large")
def create_tool_dialog():
    with open(TOOLS_FILE, 'r') as file:
        tool_code = file.read()

    with st.form("create_tool_form", border=False):
        tool_code_input = st.text_area("Tools Import and Class Definitions", value=tool_code, height=500)
        
        if st.form_submit_button("Update"):
            with open(TOOLS_FILE, 'w') as file:
                file.write(tool_code_input)
            st.toast("Tools file has been updated.")
            st.session_state.show_create_tool_dialog = False
            st.rerun()

def append_tool_to_file(tool_code):
    with open(TOOLS_FILE, 'a') as file:
        file.write(f"\n# {tool_code.split('class ')[1].split('(')[0].strip()}\n{tool_code}\n")

upload_document_placeholder = st.empty()

@st.experimental_fragment
def display_llm_settings():
    with st.expander("LLM Settings", expanded=True):
        col1, col2 = st.columns([2, 1])
        llm_options = ["OpenAi", "Anthropic", "Mistral", "Groq", "Local"]

        llm_selection_placeholder = col1.empty()
        local_model_placeholder = col2.empty()

        llm_selection_placeholder.selectbox("Select LLM Provider", options=llm_options, key='llm_model', on_change=update_model)
        if st.session_state.llm_model == "Local":
            local_providers = sorted([provider for provider in MODEL_SETTINGS.keys() if is_local_model(provider)])
            # Initialize the default provider from the environment variable
            model_name = os.getenv("OPENAI_MODEL_NAME")
            default_provider = next((key for key, value in MODEL_SETTINGS.items() if value["OPENAI_MODEL_NAME"] == model_name), local_providers[0])
            st.session_state.local_model = default_provider
            local_model_placeholder.selectbox("Provider", options=local_providers, index=local_providers.index(default_provider), key='local_model', on_change=update_model)

        st.text_input("API Base", value=st.session_state.api_base, key='api_base', on_change=update_model_settings_field)
        st.text_input("Model", value=st.session_state.model_name, key='model_name', on_change=update_model_settings_field)
        st.text_input("API Key", value=st.session_state.api_key, key='api_key', type='password', on_change=update_model_settings_field)

        if st.button(":heavy_multiplication_x: Clear Chat"):
            clear_conversation_history()
            st.rerun()

        st.toggle("Enable TTS", value=True, help="Uses Edge-TTS", key="enable_tts")

if st.session_state.llm_model == "Groq":
    uploaded_file = st.file_uploader("Upload Documents", label_visibility="collapsed", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"], key=st.session_state.widget_key)
    if uploaded_file is not None:
        with st.chat_message("assistant"):
            with st.spinner("Transcribing..."):
                transcription = transcribe_audio(uploaded_file)
                with st.expander("Transcription Details", expanded=False):
                    save_transcription_to_file(transcription, uploaded_file.name)
                    st.session_state.messages.append({"role": "assistant", "content": transcription.text})
                    save_conversation_history(st.session_state.messages)

                    st.session_state.widget_key = str(randint(1000, 100000000))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

placeholder_response = st.empty()
pleaceholder_response2 = st.empty()

if prompt := st.chat_input("Type your message here...", key="prompt"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_conversation_history(st.session_state.messages)

    agent = st.session_state.get("agent", "")
    framework = st.session_state.get("framework", "none")
    if agent == "Auto Generate New Agents":
        prompt = st.session_state.prompt
        framework = st.session_state.get("framework", "none")
        config_list = st.session_state.config_list

        if framework.lower() == "battle (run crewai & autogen)":
            with st.chat_message("assistant"):
                with st.spinner("Generating CrewAi Response..."):
                    response = generate_response("CrewAi", prompt, agent)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            with st.chat_message("assistant"):
                with st.spinner("Generating AutoGen Response..."):
                    response = generate_response("AutoGen", prompt, agent)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.chat_message("assistant"):
                with st.spinner(f"Generating {framework} Response..."):
                    response = generate_auto_response(framework, prompt, config_list)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    elif framework == "Open Interpreter":
        with st.chat_message("assistant"):
            with st.spinner(f"Generating {framework} Response..."):
                response = generate_open_interpreter_response(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            response = generate_llm_response()
            st.session_state.messages.append({"role": "assistant", "content": response})
    save_conversation_history(st.session_state.messages)

    VOICE = "en-GB-SoniaNeural"
    OUTPUT_FILE = "tts_output.mp3"

    if st.session_state.enable_tts:
        asyncio.run(synthesize_text_to_speech(response, VOICE, OUTPUT_FILE))
        st.audio(OUTPUT_FILE, autoplay=True)


def display_agent_settings():
    with st.expander("Agent Settings", expanded=True):

        if st.button(":wrench: Modify Tools"):
            st.session_state.show_create_tool_dialog = True
            create_tool_dialog()
        framework = st.selectbox("Agentic Framework", options=FRAMEWORK_OPTIONS, index=FRAMEWORK_OPTIONS.index(DEFAULT_FRAMEWORK), key="framework")

        if framework in ["AutoGen", "CrewAi", "Battle (Run CrewAi & AutoGen)"]:
            
            placeholder_response.empty()
            pleaceholder_response2.empty()

            agent = st.selectbox("Select Existing Agents", options=["Auto Generate New Agents"] + sorted(get_agents_list()), index=0, key="agent")
            if agent != "Auto Generate New Agents":
                agents_col1, agents_col2 = st.columns(2, gap="small")
                with agents_col1:
                    if st.button(":white_check_mark: Run", use_container_width=True):
                        prompt = st.session_state.prompt
                        framework = st.session_state.get("framework", "none")
                        agent = st.session_state.get("agent", "")

                        if framework.lower() == "battle (run crewai & autogen)":
                            with placeholder_response.chat_message("assistant"):
                                with st.spinner("Generating CrewAi Response..."):
                                    response = generate_response("CrewAi", prompt, agent)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            with pleaceholder_response2.chat_message("assistant"):
                                with st.spinner("Generating AutoGen Response..."):
                                    response = generate_response("AutoGen", prompt, agent)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            with placeholder_response.chat_message("assistant"):
                                with st.spinner(f"Generating {framework} Response..."):
                                    response = generate_response(framework, prompt, agent)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                        save_conversation_history(st.session_state.messages)
                with agents_col2:
                    if st.button(":robot_face: Modify Agents", use_container_width=True):
                        edit_agent_dialog()
        else:
            agent = None

        if framework in ["CrewAi", "Battle (Run CrewAi & AutoGen)"]:
            st.warning("Anthropic currently unsupported for CrewAi and Battle (Run CrewAi & AutoGen).")
        if framework == "Open Interpreter":
            st.warning("Open Interpreter currently unsupported for Anthropic.")
        if st.session_state.get("agent") == "Auto Generate New Agents":
            st.warning("Auto Generate New Agents currently unsupported for Anthropic.")

def display_documents_in_sidebar():
    documents = list_documents()
    if documents:
        with st.expander("Transcriptions", expanded=True):
            for document in documents:
                if st.button(document):
                    filepath = os.path.join("documents", document)
                    document_content = load_document_content(filepath)
                    st.session_state.selected_document = {
                        "name": document,
                        "content": document_content,
                        "path": filepath
                    }
                    show_document_content_dialog()

@st.experimental_dialog("Modify Transcript", width="large")
def show_document_content_dialog():
    if "selected_document" in st.session_state:
        document = st.session_state.selected_document
        with st.form(key='document_form'):
            st.text_area("Transcript Content", value=document["content"], key="document_content")
            if st.form_submit_button("Update Transcript"):
                new_content = st.session_state.document_content
                update_document_content(document["path"], new_content)
                st.success("Document updated successfully")
                st.rerun()

with st.sidebar:
    st.title("PraisonAi Chatbot")
    display_llm_settings()
    display_agent_settings()
    display_documents_in_sidebar()