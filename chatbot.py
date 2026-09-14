"""
Simple conversational chatbot — LangChain + Hugging Face (router.huggingface.co)

Concepts used (on purpose, to learn them):
  1. A basic chain:      prompt -> llm -> output_parser
  2. A parallel chain:   RunnableParallel runs two independent chains at the
                          same time and merges their results into one dict.

Keeps the conversation history in memory so follow-up questions work.
"""

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not set. Add it to your .env file, e.g.\n"
        "  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
        "Get one at https://huggingface.co/settings/tokens"
    )

# --- The model -------------------------------------------------------------
# Hugging Face exposes an OpenAI-compatible router, so we can plug it
# straight into LangChain's ChatOpenAI client by pointing base_url at it.
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3.1:novita",
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
    temperature=0.7,
)

# --- Chain 1: the actual reply (basic chain) --------------------------------
reply_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly, concise conversational assistant. "
                    "Reply naturally, like a chat, in 1 sentence."),
        MessagesPlaceholder("history"),
        ("human", "{message}"),
    ]
)
reply_chain = reply_prompt | llm | StrOutputParser()

# --- Chain 2: a one-word mood tag for the user's message (runs in parallel) -
mood_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Classify the emotional tone of the user's message in "
                    "exactly ONE word (e.g. happy, curious, frustrated, "
                    "neutral, excited, sad). Reply with only that word."),
        ("human", "{message}"),
    ]
)
mood_chain = mood_prompt | llm | StrOutputParser()

# --- Combine them: RunnableParallel fires both chains at the same time -----
chat_chain = RunnableParallel(reply=reply_chain, mood=mood_chain)


def chat(message: str, history: list) -> dict:
    """Run one turn of the chatbot. Returns {"reply": ..., "mood": ...}."""
    return chat_chain.invoke({"message": message, "history": history})


FEEDBACK_FILE = "feedback.json"


def remember(entry: dict):
    """Save one feedback entry to our simple local memory file."""
    history = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE) as f:
            history = json.load(f)
    history.append(entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(history, f, indent=2)


def collect_feedback(user_input: str, result: dict, history: list) -> str:
    """Ask how the reply landed, improve it on the spot if it didn't, and
    return the reply that should actually go into memory."""
    liked = input("Was that a good response? (y/n): ").strip().lower()

    if liked == "y":
        rating = input("Rate it 1-5: ").strip()
        remember({"message": user_input, "reply": result["reply"],
                   "good": True, "rating": rating})
        return result["reply"]

    reason = input("What was wrong with it?: ").strip()
    better = reply_chain.invoke({
        "history": history,
        "message": f"{user_input}\n\n(Your last reply to this was bad "
                   f"because: {reason}. Give a better reply.)"
    })
    print(f"Bot (improved): {better.strip()}\n")
    remember({"message": user_input, "reply": result["reply"],
               "good": False, "reason": reason, "improved_reply": better})
    return better


def main():
    print("Chatbot ready! Type 'exit' or 'quit' to stop.\n")
    history = []
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bot: Bye! 👋")
            break

        result = chat(user_input, history)
        print(f"Bot [{result['mood'].strip()}]: {result['reply'].strip()}\n")
        final_reply = collect_feedback(user_input, result, history)

        history.append(HumanMessage(user_input))
        history.append(AIMessage(final_reply))


if __name__ == "__main__":
    main()
