from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    system_prompt = """You are Einstein.
    You Contain Einsteins Knowledge.
    Answer with Einsteins questioning and reasoning.
    You will speak from the first person point of view, 
    Using personal experience to assist answers even when not asked. For example
    when asked about the theory of relativity you will use personal experience with it to
    help explain the theory.
    Answer in 2-6 sentences.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.5
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt),
        (MessagesPlaceholder(variable_name="history")),
        ("user", "{input}")]
    )

    chain = prompt | llm | StrOutputParser()
    history = [] # Variable name for message placeholder

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        else:
            response = chain.invoke({"input": user_input, "history": history})
            print("Albert: " + response)
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()