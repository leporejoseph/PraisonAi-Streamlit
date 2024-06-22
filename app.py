# app.py

from praisonai import PraisonAI
from praisonai.agents_generator import AgentsGenerator
from praisonai.auto import AutoGenerator
import anthropic
import streamlit as st
import os
import time
from utils import (
    update_env, get_agents_list, get_api_key, initialize_env, load_yaml, 
    save_yaml, initialize_session_state, save_conversation_history, clear_conversation_history, 
    save_selected_llm_provider, load_tools_from_file, edit_tool_in_file, 
    delete_tool_from_file, load_tool_class_definition, is_local_model
)
from config import MODEL_SETTINGS, FRAMEWORK_OPTIONS, DEFAULT_FRAMEWORK, AGENTS_DIR, TOOLS_FILE, AVAILABLE_TOOLS

# Set Streamlit to wide mode
st.set_page_config(
    layout="wide",
    page_title="PraisonAI Chatbot",
    page_icon=":material/robot:"
)

# Initialize the .env file with default values if not present
initialize_env()
initialize_session_state()

def update_model():
    model_name = st.session_state.llm_model
    if model_name == "Local":
        model_name = st.session_state.local_model
        save_selected_llm_provider("Local")  # Save "Local" instead of the specific local model name
    else:
        save_selected_llm_provider(model_name)

    st.session_state.api_key = get_api_key(model_name)
    update_env(model_name, st.session_state.api_base, st.session_state.api_key)

def generate_response(framework_name, prompt, agent):
    config_list = [
        {
            'model': st.session_state.model_name,
            'base_url': st.session_state.api_base,
            'api_key': st.session_state.api_key
        }
    ]

    if agent == "Auto Generate New Agents":
        existing_files = [f for f in os.listdir(AGENTS_DIR) if f.endswith('.yaml')]
        new_file_number = len(existing_files) + 1
        agent_file_path = f"{AGENTS_DIR}/Agents_{new_file_number}.yaml"

        generator = AutoGenerator(topic=prompt, agent_file=agent_file_path, framework=framework_name)
        agent_file = generator.generate()
        agents_generator = AgentsGenerator(agent_file, framework_name, config_list)
        response = agents_generator.generate_crew_and_kickoff()
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_conversation_history(st.session_state.messages)
        return response
    
    else:
        agent_file_path = f"{AGENTS_DIR}/{agent}"

        # Call PraisonAI and get the results
        praison_ai = PraisonAI(agent_file=agent_file_path, framework=framework_name)
        praison_ai_result = praison_ai.main()

        # Find the latest user message
        for message in reversed(st.session_state.messages):
            if message['role'] == 'user':
                message['content'] += f"\nContext:\n{praison_ai_result}"
                break

        if st.session_state.llm_model.lower() == "anthropic":
            stream = st.session_state.client.messages.stream(
                model=st.session_state["model_name"],
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                max_tokens=1024,
            )
            response = ""
            for text in stream.text_stream:
                response += text
        else:
            stream = st.session_state.client.chat.completions.create(
                model=st.session_state["model_name"],
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            )
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})
        save_conversation_history(st.session_state.messages)

        return response

@st.experimental_dialog("Edit Agents", width="large")
def edit_agent_dialog():
    yaml_data = load_yaml(f"{AGENTS_DIR}/{agent}")
    roles = yaml_data.get("roles", {})
    updated_roles = {}

    # Load custom tools
    custom_tools = load_tools_from_file(TOOLS_FILE).keys()
    all_tools = sorted(set(AVAILABLE_TOOLS + list(custom_tools)))

    topic = st.text_area("Topic", value=yaml_data.get("topic", ""), height=150)

    col1, col2 = st.columns(2)
    col_idx = 0

    for role_name, role_data in roles.items():
        col = [col1, col2][col_idx % 2]
        col_idx += 1
        with col.container(border=True):
            st.subheader(f"{role_data.get('role', '')}")
            default_tools = [tool for tool in role_data.get("tools", []) if tool in all_tools]
            tools = st.multiselect("Tools", options=all_tools, default=default_tools, key=f"tools_{role_name}")
            role = st.text_input("Role", value=role_data.get("role", ""), key=f"role_{role_name}")
            backstory = st.text_area("Backstory", value=role_data.get("backstory", ""), key=f"backstory_{role_name}")
            goal = st.text_area("Goal", value=role_data.get("goal", ""), key=f"goal_{role_name}")

            updated_tasks = {
                task_name: {
                    "description": st.text_area(f"Description: {task_name.replace('_', ' ').title()}", value=task_data.get("description", ""), key=f"description_{role_name}_{task_name}"),
                    "expected_output": st.text_area(f"Expected Output: {task_name.replace('_', ' ').title()}", value=task_data.get("expected_output", ""), key=f"expected_output_{role_name}_{task_name}")
                } for task_name, task_data in role_data.get("tasks", {}).items()
            }

            updated_roles[role_name] = {
                "role": role,
                "backstory": backstory,
                "goal": goal,
                "tasks": updated_tasks,
                "tools": tools
            }

    if st.button("Update"):
        yaml_data["topic"] = topic
        yaml_data["roles"] = updated_roles
        save_yaml(yaml_data, f"{AGENTS_DIR}/{agent}")
        st.toast(f":robot: {agent} has been updated.")
        st.session_state.show_edit_container = False
        st.rerun()

    if st.button("Cancel"):
        st.session_state.show_edit_container = False
        st.rerun()

@st.experimental_dialog("Create New Tool", width="large")
def create_tool_dialog():
    class_definition = st.text_area("Tool Class Definition", height=500)

    if st.button("Save"):
        if class_definition:
            tool_name = class_definition.split('class ')[1].split('(')[0].strip()
            tool_code = f"# {tool_name}\n{class_definition}\n# {tool_name}"
            append_tool_to_file(tool_code)
            st.toast(f"Tool '{tool_name}' has been created.")
            st.session_state.tools[tool_name] = tool_code
            st.session_state.show_create_tool_dialog = False
            st.rerun()

    if st.button("Cancel"):
        st.session_state.show_create_tool_dialog = False
        st.rerun()

def append_tool_to_file(tool_code):
    with open(TOOLS_FILE, 'a') as file:
        file.write(f"\n{tool_code}\n")

@st.experimental_dialog("Edit Tool", width="large")
def edit_tool_dialog(tool_name):
    class_definition = load_tool_class_definition(tool_name)
    start_comment = f"# {tool_name}"
    end_comment = f"# {tool_name}"
    start_idx = class_definition.find(start_comment) + len(start_comment)
    end_idx = class_definition.find(end_comment, start_idx)
    code_to_edit = class_definition[start_idx:end_idx].strip()

    updated_class_definition = st.text_area("Tool Class Definition", value=code_to_edit, height=500)

    if st.button("Save"):
        new_tool_code = f"{start_comment}\n{updated_class_definition}\n{end_comment}"
        edit_tool_in_file(tool_name, new_tool_code)
        st.toast(f"Tool '{tool_name}' has been updated.")
        st.session_state.show_edit_tool_dialog = False
        st.rerun()

    if st.button("Delete"):
        delete_tool_from_file(tool_name)
        st.toast(f"Tool '{tool_name}' has been deleted.")
        st.session_state.show_edit_tool_dialog = False
        st.rerun()

    if st.button("Cancel"):
        st.session_state.show_edit_tool_dialog = False
        st.rerun()

# Sidebar Configuration
with st.sidebar:
    st.title("PraisonAI Chatbot")

    with st.expander("LLM Settings", expanded=True):
        col1, col2 = st.columns([2, 1])

        # Create LLM options dynamically
        llm_providers = list(MODEL_SETTINGS.keys())
        local_options = sorted([provider for provider in llm_providers if is_local_model(provider)])
        llm_options = ["OpenAi", "Anthropic", "Mistral", "Groq", "Local"]

        llm_selection_placeholder = col1.empty()
        local_model_placeholder = col2.empty()

        llm_selection_placeholder.selectbox(
            "Select LLM Provider",
            options=llm_options,
            key='llm_model',
            on_change=update_model
        )

        if st.session_state.llm_model == "Local":
            local_model_placeholder.selectbox(
                "Provider",
                options=local_options,
                index=local_options.index("LM Studio"),  # Default selection
                key='local_model',
                on_change=update_model
            )
            selected_model = st.session_state.local_model
        else:
            selected_model = st.session_state.llm_model

        # Handle KeyError when model is not found in MODEL_SETTINGS
        model_settings = MODEL_SETTINGS.get(selected_model, MODEL_SETTINGS["OpenAi"])

        st.text_input("API Base", value=model_settings["OPENAI_API_BASE"], key='api_base')
        st.text_input("Model", value=model_settings["OPENAI_MODEL_NAME"], key='model_name')
        st.text_input("API Key", value=st.session_state.api_key, key='api_key', type="password")

        if st.button("Clear Chat"):
            clear_conversation_history()

    with st.expander("Agent Settings", expanded=True):
        framework = st.selectbox("Agentic Framework", options=FRAMEWORK_OPTIONS, index=FRAMEWORK_OPTIONS.index(DEFAULT_FRAMEWORK))
        if framework != "None":
            agents_list = get_agents_list()
            agent = st.selectbox("Select Existing Agents", options=sorted(agents_list), index=agents_list.index("Auto Generate New Agents") if "Auto Generate New Agents" in agents_list else 0)

            edit_button_placeholder = st.empty()
            if agent != "Auto Generate New Agents" and not st.session_state.show_edit_container:
                with edit_button_placeholder:
                    if st.button("Edit Agent"):
                        st.session_state.show_edit_container = True
                        edit_button_placeholder.empty()
                        edit_agent_dialog()
        else:
            agent = None

    with st.expander("Custom Tools", expanded=True):
        if st.button("Create New Tool"):
            st.session_state.show_create_tool_dialog = True

        tools = load_tools_from_file(TOOLS_FILE)
        for tool_name in sorted(tools.keys()):
            if st.button(tool_name.replace('_', ' ')):
                st.session_state.selected_tool = tool_name
                st.session_state.show_edit_tool_dialog = True

# Handle dialog state outside of sidebar to ensure it runs in the main context
if 'show_edit_tool_dialog' in st.session_state and st.session_state.show_edit_tool_dialog:
    edit_tool_dialog(st.session_state.selected_tool)

if 'show_create_tool_dialog' in st.session_state and st.session_state.show_create_tool_dialog:
    create_tool_dialog()

selected_model_name = st.session_state.get('local_model') if st.session_state.llm_model == "Local" else st.session_state.llm_model
update_env(selected_model_name, st.session_state.api_base, st.session_state.api_key)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_conversation_history(st.session_state.messages)

    with st.chat_message("assistant"):
        if framework.lower() == "none":
            if st.session_state.llm_model.lower() == "anthropic":
                client = anthropic.Anthropic()

                stream_response = st.empty()
                with client.messages.stream(
                    max_tokens=1024,
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    model=st.session_state["model_name"],
                ) as stream:
                    response = ""
                    for text in stream.text_stream:
                        response += text
                        stream_response.write(response)
            else:
                stream = st.session_state.client.chat.completions.create(
                    model=st.session_state["model_name"],
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    stream=True,
                )
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        elif framework.lower() == "battle":
            with st.spinner("Generating CrewAi Response..."):
                response_crewai = generate_response("crewai", prompt, agent)
                st.session_state.messages.append({"role": "assistant", "content": response_crewai})

            with st.spinner("Generating AutoGen Response..."):
                response_autogen = generate_response("autogen", prompt, agent)
                st.session_state.messages.append({"role": "assistant", "content": response_autogen})
        else:
            with st.spinner("Generating response..."):
                response = generate_response(framework.lower(), prompt, agent)

        save_conversation_history(st.session_state.messages)
        st.rerun()