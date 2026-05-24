import streamlit as st

from tools import vector_store
from agent import agent_app

st.set_page_config(page_title="Small Language Model", layout="centered", page_icon="🤖")
st.title("Botro Chatbot")
st.caption("Localised chatbot with continuous memory synching")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph_messages" not in st.session_state:
    st.session_state.graph_messages = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    with st.status("Checking long-term vector database...", expanded=False) as status:
        cached_docs = vector_store.similarity_search(user_prompt, k=1)

        injected_prompt = user_prompt
        if cached_docs:
            status.update(label="Local memory match found! Injecting context.", state="complete")
            injected_prompt = (
                f"Context from your local long-term memory database:\n"
                f"{cached_docs[0].page_content}\n\n"
                f"User Question: {user_prompt}\n"
                f"Instructions: Use the provided memory context above if it answers the question. "
                f"If the context is out-of-date, use your search tool."
            )
        else:
            status.update(label="No matching local memory found. Querying LLM directly.", state="complete")

    st.session_state.graph_messages.append(("user", injected_prompt))

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        with st.status("Thinking...", expanded=True) as agent_status:
            inputs = {"messages": st.session_state.graph_messages}
            final_text_output = ""

            for event in agent_app.stream(inputs, stream_mode="values"):
                last_msg = event["messages"][-1]

                if last_msg.type == "ai" and last_msg.tool_calls:

                    first_call = last_msg.tool_calls[0]

                    tool_args = first_call.get('args', {})

                    query_text = tool_args.get('query') if tool_args else None

                    if not query_text:
                        query_text = "Analyzing search parameters..."

                    agent_status.update(label=f"Tool Requested: Searching Web for '{query_text}'...")

                if last_msg.type == "ai" and not last_msg.tool_calls:
                    agent_status.update(label="Final response compiled!", state="complete")
                    response_placeholder.markdown(last_msg.content)
                    final_text_output = last_msg.content

            st.session_state.graph_messages = event["messages"]
            st.session_state.chat_history.append({"role": "assistant", "content": final_text_output})