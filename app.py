import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.7,
)

SYSTEM_PROMPT = "You are a helpful, friendly AI assistant. Answer clearly and concisely."


def _text(content):
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def chat_fn(message, history):
    # history Gradio se aati hai, isliye har user ki chat alag rehti hai
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        if isinstance(turn, dict):
            role, content = turn["role"], _text(turn["content"])
            msgs.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
        else:
            user, bot = turn
            msgs.append(HumanMessage(content=_text(user)))
            if bot:
                msgs.append(AIMessage(content=_text(bot)))
    msgs.append(HumanMessage(content=_text(message)))

    try:
        return llm.invoke(msgs).content
    except Exception as e:
        return f"Error calling Groq API: {type(e).__name__}: {e}"


demo = gr.ChatInterface(
    fn=chat_fn,
    title="✨ Spark AI",
    examples=["Hello, who are you?", "Explain quantum computing simply."],
)

if __name__ == "__main__":
    demo.launch()