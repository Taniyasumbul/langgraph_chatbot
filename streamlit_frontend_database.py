import streamlit as st
from langgraph_database_backend import chatbot, retrieve_all_threads, get_available_tools
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import json
from datetime import datetime
import sqlite3

# **************************************** Page Config *************************
st.set_page_config(
    page_title="AI Agent Chatbot",
    
    layout="wide",
    initial_sidebar_state="expanded"
)

# **************************************** Utility Functions *************************

def generate_thread_id():
    """Generate a unique thread ID."""
    return str(uuid.uuid4())

def reset_chat():
    """Create a new chat thread."""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []
    st.session_state['chat_metadata'][thread_id] = {
        'created_at': datetime.now().isoformat(),
        'title': 'New Chat',
        'message_count': 0
    }

def add_thread(thread_id):
    """Add thread to list if not already present."""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        if thread_id not in st.session_state['chat_metadata']:
            st.session_state['chat_metadata'][thread_id] = {
                'created_at': datetime.now().isoformat(),
                'title': 'New Chat',
                'message_count': 0
            }

def load_conversation(thread_id):
    """Load conversation history for a specific thread."""
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        return state.values.get('messages', [])
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        return []

def format_thread_name(thread_id, messages):
    """Create a readable name for the thread button."""
    if not messages:
        return "New Chat"
    
    # Get first user message
    for msg in messages:
        if isinstance(msg, HumanMessage):
            preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            return preview
    
    return "Empty Chat"

def delete_thread(thread_id):
    """Delete a thread from the list."""
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
    if thread_id in st.session_state['chat_metadata']:
        del st.session_state['chat_metadata'][thread_id]

def export_conversation(thread_id):
    """Export conversation to JSON format."""
    messages = load_conversation(thread_id)
    export_data = {
        'thread_id': thread_id,
        'exported_at': datetime.now().isoformat(),
        'messages': []
    }
    
    for msg in messages:
        export_data['messages'].append({
            'role': 'user' if isinstance(msg, HumanMessage) else 'assistant',
            'content': msg.content,
            'timestamp': getattr(msg, 'timestamp', None)
        })
    
    return json.dumps(export_data, indent=2)

def search_conversations(query):
    """Search through all conversations for matching content."""
    results = []
    query_lower = query.lower()
    
    for thread_id in st.session_state['chat_threads']:
        messages = load_conversation(thread_id)
        for msg in messages:
            if query_lower in msg.content.lower():
                results.append({
                    'thread_id': thread_id,
                    'preview': format_thread_name(thread_id, messages),
                    'match': msg.content[:100]
                })
                break
    
    return results

def update_chat_metadata(thread_id):
    """Update metadata for a chat thread."""
    messages = load_conversation(thread_id)
    if thread_id not in st.session_state['chat_metadata']:
        st.session_state['chat_metadata'][thread_id] = {}
    
    st.session_state['chat_metadata'][thread_id].update({
        'message_count': len(messages),
        'last_updated': datetime.now().isoformat(),
        'title': format_thread_name(thread_id, messages)
    })

# **************************************** Session Setup ******************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'chat_metadata' not in st.session_state:
    st.session_state['chat_metadata'] = {}

if 'search_mode' not in st.session_state:
    st.session_state['search_mode'] = False

if 'show_settings' not in st.session_state:
    st.session_state['show_settings'] = False

if 'system_prompt' not in st.session_state:
    st.session_state['system_prompt'] = ""

if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0.7

if 'show_tool_details' not in st.session_state:
    st.session_state['show_tool_details'] = True

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************

with st.sidebar:
    st.title(' AI Agent Chatbot')
    st.caption('Powered by LangGraph ')
    
    # New Chat Button
    if st.button('New Chat', use_container_width=True, type='primary'):
        reset_chat()
        st.rerun()
    
    st.divider()
    
    # Quick Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Stats', use_container_width=True):
            st.session_state['show_stats'] = not st.session_state.get('show_stats', False)
    with col2:
        if st.button(' Settings', use_container_width=True):
            st.session_state['show_settings'] = not st.session_state.get('show_settings', False)
    
    # Settings Panel
    if st.session_state.get('show_settings', False):
        with st.expander(" Agent Settings", expanded=True):
            st.session_state['temperature'] = st.slider(
                'Temperature',
                min_value=0.0,
                max_value=1.0,
                value=st.session_state['temperature'],
                step=0.1,
                help='Higher = more creative, Lower = more focused'
            )
            
            st.session_state['system_prompt'] = st.text_area(
                'System Prompt',
                value=st.session_state['system_prompt'],
                placeholder='Enter custom instructions for the AI...',
                height=100
            )
            
            st.session_state['show_tool_details'] = st.checkbox(
                'Show Tool Details',
                value=st.session_state['show_tool_details'],
                help='Display tool usage in chat'
            )
            
            if st.button('Clear All Chats', type='secondary', use_container_width=True):
                if st.checkbox(' Confirm deletion'):
                    st.session_state['chat_threads'] = [st.session_state['thread_id']]
                    st.session_state['chat_metadata'] = {}
                    st.success('All chats cleared!')
                    st.rerun()
    
    # Available Tools
    with st.expander(" Available Tools", expanded=False):
        st.caption('The agent can use these tools automatically:')
        tools_info = get_available_tools()
        
        tool_categories = {
            ' Web & Data': ['search_wikipedia', 'web_scraper', 'api_caller'],
            ' Processing': ['calculate', 'data_processor', 'python_code_executor'],
            ' Analysis': ['text_analyzer', 'analyze_json', 'file_analyzer', 'regex_matcher'],
            ' Creation': ['create_chart_data'],
            ' Utilities': ['get_current_time']
        }
        
        for category, tool_names in tool_categories.items():
            st.markdown(f"**{category}**")
            for tool in tools_info:
                if tool['name'] in tool_names:
                    with st.container():
                        st.markdown(f"• `{tool['name']}`")
                        st.caption(tool['description'])
            st.divider()
    
    # Search Conversations
    st.subheader(' Search Chats')
    search_query = st.text_input('Search...', placeholder='Type to search', label_visibility='collapsed')
    
    if search_query:
        results = search_conversations(search_query)
        if results:
            st.success(f'Found {len(results)} results')
            for result in results[:5]:
                if st.button(
                    f" {result['preview'][:35]}...",
                    key=f"search_{result['thread_id']}",
                    use_container_width=True
                ):
                    st.session_state['thread_id'] = result['thread_id']
                    messages = load_conversation(result['thread_id'])
                    temp_messages = []
                    for msg in messages:
                        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                        temp_messages.append({'role': role, 'content': msg.content})
                    st.session_state['message_history'] = temp_messages
                    st.rerun()
        else:
            st.info('No results found')
    
    st.divider()
    
    # My Conversations
    st.subheader(' My Conversations')
    
    # Sort options
    sort_option = st.selectbox(
        'Sort by',
        ['Recent', 'Oldest', 'Most Messages'],
        label_visibility='collapsed'
    )
    
    # Sort threads
    sorted_threads = st.session_state['chat_threads'].copy()
    if sort_option == 'Recent':
        sorted_threads = sorted_threads[::-1]
    elif sort_option == 'Most Messages':
        sorted_threads = sorted(
            sorted_threads,
            key=lambda t: st.session_state['chat_metadata'].get(t, {}).get('message_count', 0),
            reverse=True
        )
    
    # Display threads
    if not sorted_threads:
        st.info('No conversations yet. Start chatting!')
    
    for thread_id in sorted_threads:
        messages = load_conversation(thread_id)
        update_chat_metadata(thread_id)
        
        metadata = st.session_state['chat_metadata'].get(thread_id, {})
        button_label = metadata.get('title', format_thread_name(thread_id, messages))
        message_count = metadata.get('message_count', len(messages))
        
        # Highlight current thread
        is_current = thread_id == st.session_state['thread_id']
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            if st.button(
                f"{' ' if is_current else ''}  {button_label}",
                key=f"thread_{thread_id}",
                use_container_width=True,
                type='primary' if is_current else 'secondary'
            ):
                st.session_state['thread_id'] = thread_id
                temp_messages = []
                for msg in messages:
                    role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                    temp_messages.append({'role': role, 'content': msg.content})
                st.session_state['message_history'] = temp_messages
                st.rerun()
            
            st.caption(f" {message_count} messages")
        
        with col2:
            # More options
            with st.popover("⋮"):
                if st.button("Export", key=f"export_{thread_id}", use_container_width=True):
                    export_data = export_conversation(thread_id)
                    st.download_button(
                        label="Download",
                        data=export_data,
                        file_name=f"chat_{thread_id[:8]}.json",
                        mime="application/json",
                        key=f"download_{thread_id}"
                    )
                
                if st.button(" Delete", key=f"delete_{thread_id}", use_container_width=True):
                    if thread_id == st.session_state['thread_id']:
                        reset_chat()
                    delete_thread(thread_id)
                    st.rerun()
    
    st.divider()
    
    # Footer
    st.caption(' **Quick Tips:**')
    st.caption('• Agent uses tools automatically')
    st.caption('• Try: "Read https://example.com"')
    st.caption('• Try: "Sort: 5, 2, 8, 1, 9"')
    st.caption('• Try: "Search Wikipedia for AI"')

# **************************************** Main UI ************************************

# Header
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.title(' AI Agent Assistant')
with col2:
    st.metric("Chats", len(st.session_state['chat_threads']))
with col3:
    st.metric("Messages", len(st.session_state['message_history']))

st.divider()


# Display conversation history
for idx, message in enumerate(st.session_state['message_history']):
    with st.chat_message(message['role']):
        # Show tool usage if present and enabled
        if st.session_state['show_tool_details'] and message.get('tool_calls'):
            with st.expander(" Tools Used", expanded=False):
                for tool_call in message['tool_calls']:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    
                    st.code(f"Tool: {tool_name}", language='text')
                    st.json(tool_args)
        
        # Show tool results if present
        if st.session_state['show_tool_details'] and message.get('tool_result'):
            with st.expander(" Tool Results", expanded=False):
                st.text(message['tool_result'])
        
        # Main message content
        st.markdown(message['content'])
        
        # Action buttons for assistant messages
        if message['role'] == 'assistant':
            col1, col2, col3, col4 = st.columns([1, 1, 1, 10])
            with col1:
                if st.button("", key=f"copy_{idx}", help="Copy"):
                    st.toast("Copied!")
            with col2:
                if st.button("", key=f"regen_{idx}", help="Regenerate"):
                    st.session_state['message_history'] = st.session_state['message_history'][:idx]
                    st.rerun()
            with col3:
                if st.button("", key=f"like_{idx}", help="Like"):
                    st.toast("Thanks for feedback!")

# Chat input
user_input = st.chat_input('Ask me anything... I can use tools to help you!')

# Handle example query
if 'example_query' in st.session_state:
    user_input = st.session_state['example_query']
    del st.session_state['example_query']

if user_input:
    # Add user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    
    with st.chat_message('user'):
        st.markdown(user_input)
    
    # Configuration
    config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }
    
    # Prepare messages
    messages_to_send = [HumanMessage(content=user_input)]
    if st.session_state['system_prompt']:
        messages_to_send.insert(0, HumanMessage(content=f"System: {st.session_state['system_prompt']}"))
    
    # Stream response
    with st.chat_message('assistant'):
        try:
            response_content = ""
            response_placeholder = st.empty()
            tool_calls_made = []
            tool_results = []
            
            # Progress indicator
            status = st.status(" Thinking...", expanded=st.session_state['show_tool_details'])
            
            # Stream the response
            for event in chatbot.stream(
                {'messages': messages_to_send},
                config=config,
                stream_mode='values'
            ):
                if 'messages' in event and len(event['messages']) > 0:
                    last_msg = event['messages'][-1]
                    
                    # Track tool calls
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            tool_name = tool_call.get('name', 'unknown')
                            tool_args = tool_call.get('args', {})
                            tool_calls_made.append({'name': tool_name, 'args': tool_args})
                            status.update(label=f"🔧 Using: {tool_name}", state="running")
                    
                    # Track tool results
                    if isinstance(last_msg, ToolMessage):
                        tool_results.append(last_msg.content)
                        status.update(label=f"Got results", state="running")
                    
                    # Display AI response
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        response_content = last_msg.content
                        status.update(label="✨ Generating response...", state="running")
            
            status.update(label="Complete!", state="complete")
            response_placeholder.markdown(response_content)
            
            # Save to history
            message_data = {
                'role': 'assistant',
                'content': response_content
            }
            
            if tool_calls_made:
                message_data['tool_calls'] = tool_calls_made
            if tool_results:
                message_data['tool_result'] = "\n".join(tool_results)
            
            st.session_state['message_history'].append(message_data)
            update_chat_metadata(st.session_state['thread_id'])
            
            # Show success message
            if tool_calls_made:
                st.success(f"Used {len(tool_calls_made)} tool(s) to answer your question!")
            
        except Exception as e:
            st.error(f" Error: {str(e)}")
            st.session_state['message_history'].append({
                'role': 'assistant',
                'content': "Sorry, I encountered an error. Please try again."
            })