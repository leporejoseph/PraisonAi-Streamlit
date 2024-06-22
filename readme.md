# PraisonAI Chatbot

PraisonAI is a sophisticated chatbot built using the Streamlit framework. It integrates multiple language model providers to offer a flexible and customizable conversational experience. This project includes tools for managing and editing chatbot agents, and can be configured to use different AI frameworks.

## Connect with Me
| Contact Info       |                      |
|--------------------|----------------------|
| Joseph LePore  | [![Linkedin Badge](https://img.shields.io/badge/-Linkedin-blue?style=flat&logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/joseph-lepore-062561b3/)    |

## Features

- **Multiple LLM Providers**: Select from various language model providers such as OpenAI, Ollama Mistral, FastChat, LM Studio, Mistral, Groq, and Anthropic.
- **Agent Management**: Create, edit, and manage multiple chatbot agents with customizable roles and tools.
- **Flexible Frameworks**: Supports different frameworks for generating responses including CrewAi, AutoGen, and Battle.
- **Interactive UI**: Streamlit-based interface with wide mode configuration for enhanced usability.
- **Custom Tools**: Create and manage custom tools to enhance chatbot capabilities.

## Installation

To set up the PraisonAI Chatbot, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/leporejoseph/praisonai-chatbot.git
    cd praisonai-chatbot
    ```

2. **Install the dependencies** (including the latest versions):
    ```sh
    pip install -r requirements.txt --upgrade
    ```

3. **Run the application**:
    ```sh
    streamlit run app.py
    ```

## Configuration

The chatbot configuration is managed through the `.env` file and Streamlit sidebar. Here are some key settings:

- **LLM Settings**: Choose the language model provider and set API keys.
- **Agent Settings**: Select or create agents, customize their roles, tools, and other properties.

### .env File

Ensure the `.env` file contains the appropriate API keys and settings for your chosen LLM provider. The file is automatically initialized with default values if not present.

## Usage

### Running the Chatbot

Once the application is running, you can interact with the chatbot through the Streamlit interface. Use the sidebar to configure LLM and agent settings.

### Editing Agents

To edit an existing agent:
1. Select the agent from the dropdown menu in the sidebar.
2. Click on "Edit Agent" to open the edit dialog.
3. Modify the agent's roles, tools, and other properties as needed.
4. Click "Update" to save the changes.

### Creating and Editing Tools

To create a new tool:
1. Click on "Create New Tool" in the sidebar.
2. Define the tool class and save it.

To edit an existing tool:
1. Select the tool from the list in the sidebar.
2. Modify the tool class definition and save it.

### Important Functions

<details>
<summary>Click to expand the list of important functions</summary>

#### app.py

- **update_model**: Updates the selected model and API key.
- **generate_response**: Generates a response using the specified framework and prompt.
- **edit_agent_dialog**: Dialog for editing an existing agent.
- **create_tool_dialog**: Dialog for creating a new tool.
- **edit_tool_dialog**: Dialog for editing an existing tool.

#### utils.py

- **initialize_env**: Initializes the environment variables.
- **update_env**: Updates the environment variables based on the selected model.
- **get_api_key**: Retrieves the API key for the selected model.
- **get_agents_list**: Retrieves the list of available agents.
- **load_yaml**: Loads a YAML file.
- **save_yaml**: Saves data to a YAML file.
- **load_conversation_history**: Loads the conversation history.
- **save_conversation_history**: Saves the conversation history.
- **clear_conversation_history**: Clears the conversation history.
- **load_selected_llm_provider**: Loads the selected LLM provider from the configuration.
- **save_selected_llm_provider**: Saves the selected LLM provider to the configuration.
- **initialize_session_state**: Initializes the session state.
- **load_tools_from_file**: Loads custom tools from a file.
- **edit_tool_in_file**: Edits an existing tool in the file.
- **load_tool_class_definition**: Loads the class definition of a tool.
- **delete_tool_from_file**: Deletes a tool from the file.

</details>

## Contribution

We welcome contributions to improve PraisonAI Chatbot. To contribute:

1. **Fork the repository**:
    ```sh
    git fork https://github.com/leporejoseph/praisonai-chatbot.git
    ```

2. **Create a new branch**:
    ```sh
    git checkout -b feature/your-feature-name
    ```

3. **Make your changes** and commit them:
    ```sh
    git commit -m "Add your message here"
    ```

4. **Push to your fork**:
    ```sh
    git push origin feature/your-feature-name
    ```

5. **Create a pull request**.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
