# app.py

from praisonai import PraisonAI
from praisonai.agents_generator import AgentsGenerator
from praisonai.inc import PraisonAIModel

import streamlit as st
import os
from utils import *
from config import *
import asyncio
from openai import OpenAI
from litellm import completion

st.set_page_config(layout="wide", page_title="PraisonAI Chatbot", page_icon=":robot_face:")

initialize_env()
initialize_session_state()

uploaded_file_placeholder = st.empty()
tool_dialog = st.empty()

@st.experimental_fragment
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
    st.session_state.config_list = [{
            'model': st.session_state.model_name,
            'base_url': st.session_state.api_base,
            'api_key': st.session_state.api_key,
            'api_type': st.session_state.llm_model.lower()
        }]

    if st.session_state.llm_model == "Groq":
        uploaded_file = uploaded_file_placeholder.file_uploader("Upload Documents", label_visibility="collapsed", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"], key=st.session_state.widget_key)
        if uploaded_file is not None:
            with st.chat_message("assistant"):
                with st.spinner("Transcribing..."):
                    transcription = transcribe_audio(uploaded_file)
                    with st.expander("Transcription Details", expanded=False):
                        save_transcription_to_file(transcription, uploaded_file.name)
                        st.session_state.messages.append({"role": "assistant", "content": transcription.text})
                        save_conversation_history(st.session_state.messages)

                        st.session_state.widget_key = str(randint(1000, 100000000))

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

def generate_auto_response(framework_name, prompt, config_list):
    praison_ai = PraisonAI(
        auto=prompt,
        framework=framework_name.lower()
    )
    response = praison_ai.main()

    # Move the created file to the agents folder and rename it
    move_and_rename_file('test.yaml', 'agents')

    response_header = f"### {framework_name} Response ###\n"
    response = response_header + response

    st.markdown(response)
    return response

def generate_praisonai_response(framework_name, agent, prompt):
    agent_file_path = f"{AGENTS_DIR}/{agent}"

    if framework_name.lower() == "crewai" and st.session_state.llm_model.lower() == "openrouter":
        update_env(f"openrouter/{st.session_state.model_name}", st.session_state.api_base, st.session_state.api_key)

    praison_ai = PraisonAI(
        agent_file=agent_file_path,
        framework=framework_name.lower(),
    )
    response = praison_ai.main()
        
    if framework_name.lower() == "crewai" and st.session_state.llm_model.lower() == "openrouter":
        update_env(st.session_state.model_name, st.session_state.api_base, st.session_state.api_key)

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
    if st.session_state.llm_model != "Local":
        model = st.session_state.llm_model.replace(" ", "").lower() + "/" + st.session_state["model_name"]
        extra_args = {}
    else:
        model = st.session_state["model_name"]
        extra_args = {
            "api_base": st.session_state.api_base,
            "custom_llm_provider": "openai"
        }

    stream = completion(
        model=model,
        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        stream=True,
        **extra_args
    )

    response_placeholder = st.empty()
    response = ""
    for chunk in stream:
        content = chunk["choices"][0]["delta"].get("content")
        if content:
            response += content
            response_placeholder.write(response)
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
            st.rerun()

def append_tool_to_file(tool_code):
    with open(TOOLS_FILE, 'a') as file:
        file.write(f"\n# {tool_code.split('class ')[1].split('(')[0].strip()}\n{tool_code}\n")

@st.experimental_fragment
def display_llm_settings():
    with st.expander("LLM Settings", expanded=True):
        col1, col2 = st.columns([2, 1])
        llm_options = ["OpenAi", "OpenRouter", "Mistral", "Groq", "Local"]

        llm_selection_placeholder = col1.empty()
        local_model_placeholder = col2.empty()

        llm_selection_placeholder.selectbox("Select LLM Provider", options=llm_options, key='llm_model', on_change=update_model)
        if st.session_state.llm_model == "Local":
            local_providers = sorted([provider for provider in MODEL_SETTINGS.keys() if is_local_model(provider)])
            model_name = os.getenv("OPENAI_MODEL_NAME")
            default_provider = next((key for key, value in MODEL_SETTINGS.items() if value["OPENAI_MODEL_NAME"] == model_name), local_providers[0])
            st.session_state.local_model = default_provider
            local_model_placeholder.selectbox("Provider", options=local_providers, key='local_model', on_change=update_model)

        st.text_input("API Base", key='api_base', on_change=update_model_settings_field)
        st.text_input("Model", key='model_name', on_change=update_model_settings_field)
        st.text_input("API Key", key='api_key', type='password', on_change=update_model_settings_field)

        ttsCol1, ttsCol2 = st.columns(2)
        with ttsCol1:
            st.toggle("Enable TTS", help="Uses Edge-TTS", key="enable_tts", on_change=lambda: save_config("enable_tts", st.session_state.enable_tts))
        with ttsCol2:
            enhance_response_placeholder = st.empty()
            if st.session_state.enable_tts:
                enhance_response_placeholder.toggle("Enhance TTS", help="Sends an additional LLM request to personalize the TTS response. Look at TTS_PERSONALITY in the config.py to customize this.", key="enhance_tts", on_change=lambda: save_config("enhance_tts", st.session_state.enhance_tts))

        if st.session_state.enable_tts:
            voices = asyncio.run(get_text_to_speech_voices())
            voice_options = {f"{voice['FriendlyName'].split('-')[-1].strip()} - {voice['Gender']}": voice['ShortName'] for voice in voices}
            default_index = list(voice_options.values()).index(st.session_state.tts_personality) if st.session_state.tts_personality in voice_options.values() else 0
            selected_voice = st.selectbox("Voices", options=list(voice_options.keys()), index=default_index, key="voice_select")
            st.session_state.tts_personality = voice_options[selected_voice]
            save_config("tts_personality", st.session_state.tts_personality)

        if st.button(":heavy_multiplication_x: Clear Chat"):
            clear_conversation_history()

@st.experimental_fragment
def output_tts(response):
    if st.session_state.enable_tts:
        
        tts_voice = st.session_state.get("tts_personality", "en-GB-SoniaNeural")
        enhance_tts = st.session_state.get("enhance_tts", False)

        if enhance_tts and len(response) > 150:
            client = OpenAI(api_key=st.session_state.api_key, base_url=st.session_state.api_base)
            completion = client.chat.completions.create(
                model=st.session_state["model_name"],
                messages=[
                    {"role": "system", "content": TTS_PERSONALITY},
                    {"role": "user", "content": f"Summarize this context. Summarize it as if you are speaking directly to me, making it clear that the summary is derived from your message. Context {response}"}
                ]
            )
            formatted_response = completion.choices[0].message.content
        else:
            formatted_response = response

        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            asyncio.run(synthesize_text_to_speech(formatted_response, tts_voice, OUTPUT_FILE))
            st.audio(OUTPUT_FILE, autoplay=True)



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

        with st.chat_message("assistant"):
            with st.spinner(f"Generating {framework} Response..."):
                response = generate_auto_response(framework, prompt, config_list)
                output_tts(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            response = generate_llm_response()
            output_tts(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    save_conversation_history(st.session_state.messages)

def display_agent_settings():
    with st.expander("Agent Settings", expanded=True):
        if st.button(":wrench: Modify Tools"):
            create_tool_dialog()

        framework = st.selectbox("Agentic Framework", options=FRAMEWORK_OPTIONS, index=FRAMEWORK_OPTIONS.index(DEFAULT_FRAMEWORK), key="framework")

        if framework in ["AutoGen", "CrewAi"]:
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

                        with placeholder_response.chat_message("assistant"):
                            with st.spinner(f"Generating {framework} Response..."):
                                response = generate_response(framework, prompt, agent)
                                output_tts(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                        save_conversation_history(st.session_state.messages)
                with agents_col2:
                    if st.button(":robot_face: Modify Agents", use_container_width=True):
                        edit_agent_dialog()
        else:
            agent = None

def display_documents_in_sidebar():
    documents = list_documents()
    if documents:
        with st.expander("Transcripts", expanded=False):
            selected_document_name = st.selectbox("Select a Transcript to Modify", ["Select a Transcript to Modify"] + documents, label_visibility="collapsed")

            if selected_document_name != "Select a Transcript to Modify":
                filepath = os.path.join("documents", selected_document_name)
                document_content = load_document_content(filepath)
                st.session_state.selected_document = {
                    "name": selected_document_name,
                    "content": document_content,
                    "path": filepath
                }
                if st.button(":microphone: Modify Transcript"):
                    show_document_content_dialog()

@st.experimental_dialog("Modify Transcript", width="large")
def show_document_content_dialog():
    if "selected_document" in st.session_state:
        document = st.session_state.selected_document
        with st.form(key='document_form'):
            st.text_area("Transcript Content", value=document["content"], key="document_content", height=500)
            if st.form_submit_button("Update Transcript"):
                new_content = st.session_state.document_content
                update_document_content(document["path"], new_content)
                st.success("Document updated successfully")
                st.rerun()

if st.session_state.llm_model == "Groq":
        uploaded_file = uploaded_file_placeholder.file_uploader("Upload Documents", label_visibility="collapsed", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"], key=st.session_state.widget_key)
        if uploaded_file is not None:
            with st.chat_message("assistant"):
                with st.spinner("Transcribing..."):
                    transcription = transcribe_audio(uploaded_file)
                    with st.expander("Transcription Details", expanded=False):
                        save_transcription_to_file(transcription, uploaded_file.name)
                        st.session_state.messages.append({"role": "assistant", "content": transcription.text})
                        save_conversation_history(st.session_state.messages)

                        st.session_state.widget_key = str(randint(1000, 100000000))

with st.sidebar:
    st.image('images/praisonai-logo-large.png', width=200)
    display_llm_settings()
    display_agent_settings()
    display_documents_in_sidebar()
