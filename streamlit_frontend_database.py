import streamlit as st
from langgraph_database_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import json
from datetime import datetime
import sqlite3

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

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************

st.sidebar.title(' LangGraph Chatbot')

# New Chat Button
if st.sidebar.button('➕ New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

# Settings Toggle
with st.sidebar.expander("⚙️ Settings", expanded=st.session_state['show_settings']):
    st.session_state['temperature'] = st.slider(
        'Temperature',
        min_value=0.0,
        max_value=1.0,
        value=st.session_state['temperature'],
        step=0.1,
        help='Higher values make output more random'
    )
    
    st.session_state['system_prompt'] = st.text_area(
        'System Prompt',
        value=st.session_state['system_prompt'],
        placeholder='Enter custom instructions for the AI...',
        height=100
    )
    
    if st.button('Clear All Conversations', type='secondary', use_container_width=True):
        if st.checkbox('Are you sure?'):
            st.session_state['chat_threads'] = [st.session_state['thread_id']]
            st.session_state['chat_metadata'] = {}
            st.rerun()

# Search Conversations
st.sidebar.header(' Search')
search_query = st.sidebar.text_input('Search conversations...', key='search_input')

if search_query:
    results = search_conversations(search_query)
    if results:
        st.sidebar.write(f"Found {len(results)} results:")
        for result in results:
            if st.sidebar.button(
                f" {result['preview'][:40]}...",
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
        st.sidebar.info("No results found")

# My Conversations
st.sidebar.header(' My Conversations')

# Sort options
sort_option = st.sidebar.selectbox(
    'Sort by',
    ['Recent', 'Oldest', 'Most Messages'],
    key='sort_option'
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
for thread_id in sorted_threads:
    messages = load_conversation(thread_id)
    update_chat_metadata(thread_id)
    
    metadata = st.session_state['chat_metadata'].get(thread_id, {})
    button_label = metadata.get('title', format_thread_name(thread_id, messages))
    message_count = metadata.get('message_count', len(messages))
    
    # Highlight current thread
    button_type = "primary" if thread_id == st.session_state['thread_id'] else "secondary"
    
    col1, col2 = st.sidebar.columns([4, 1])
    
    with col1:
        if st.button(
            f"{button_label} ({message_count})",
            key=f"thread_{thread_id}",
            use_container_width=True,
            type=button_type
        ):
            st.session_state['thread_id'] = thread_id
            temp_messages = []
            for msg in messages:
                role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})
            st.session_state['message_history'] = temp_messages
            st.rerun()
    
    with col2:
        # Delete and Export options in popover
        with st.popover("⋮"):
            if st.button(" Export", key=f"export_{thread_id}", use_container_width=True):
                export_data = export_conversation(thread_id)
                st.download_button(
                    label="Download JSON",
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

# **************************************** Main UI ************************************

st.title(' Chat with me')

# Stats bar
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Chats", len(st.session_state['chat_threads']))
with col2:
    current_messages = len(st.session_state['message_history'])
    st.metric("Messages", current_messages)
with col3:
    total_messages = sum(
        meta.get('message_count', 0) 
        for meta in st.session_state['chat_metadata'].values()
    )
    st.metric("All Messages", total_messages)

st.divider()

# Display conversation history
for idx, message in enumerate(st.session_state['message_history']):
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
        # Add copy and regenerate buttons for assistant messages
        if message['role'] == 'assistant':
            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                if st.button( key=f"copy_{idx}", help="Copy message"):
                    st.toast("Copied to clipboard!")
            with col2:
                if st.button (key=f"regen_{idx}", help="Regenerate response"):
                    # Remove this and subsequent messages
                    st.session_state['message_history'] = st.session_state['message_history'][:idx]
                    st.rerun()

# Chat input with file upload option
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.chat_input('Type your message here...')

with col2:
    uploaded_file = st.file_uploader(
        "📎",
        type=['txt', 'pdf', 'json'],
        label_visibility='collapsed',
        key='file_upload'
    )

# Process file upload
if uploaded_file:
    file_content = uploaded_file.read().decode('utf-8')
    user_input = f"[File: {uploaded_file.name}]\n\n{file_content[:1000]}"
    st.info(f"📎 Attached: {uploaded_file.name}")

if user_input:
    # Add user message to history and display
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    
    # Configuration for LangGraph
    config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }
    
    # Prepare messages with system prompt if provided
    messages_to_send = [HumanMessage(content=user_input)]
    if st.session_state['system_prompt']:
        messages_to_send.insert(0, HumanMessage(content=f"System: {st.session_state['system_prompt']}"))
    
    # Stream assistant response
    with st.chat_message('assistant'):
        try:
            response_content = ""
            response_placeholder = st.empty()
            
            # Show thinking indicator
            with st.spinner('Thinking...'):
                # Stream the response
                for chunk, metadata in chatbot.stream(
                    {'messages': messages_to_send},
                    config=config,
                    stream_mode='messages'
                ):
                    # Filter for AIMessage chunks with content
                    if hasattr(chunk, 'content') and chunk.content:
                        response_content += chunk.content
                        response_placeholder.markdown(response_content)
            
            # Add to message history
            st.session_state['message_history'].append({
                'role': 'assistant',
                'content': response_content
            })
            
            # Update metadata
            update_chat_metadata(st.session_state['thread_id'])
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.session_state['message_history'].append({
                'role': 'assistant',
                'content': "Sorry, I encountered an error. Please try again."
            })

# Footer with keyboard shortcuts
st.sidebar.divider()
st.sidebar.caption(" **Tips:**")
st.sidebar.caption("• Use search to find old chats")
st.sidebar.caption("• Click ⋮ to export/delete chats")
st.sidebar.caption("• Adjust temperature for creativity")
st.sidebar.caption("• Upload files to analyze them")