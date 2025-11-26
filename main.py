from dotenv import load_dotenv
import os
import gradio as gr

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

    # General Setup for our llm
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.5
    )

    # Seting up our prompt to be sent to the chatbot.
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt),
        (MessagesPlaceholder(variable_name="history")),
        ("user", "{input}")]
    )

    chain = prompt | llm | StrOutputParser()

    def chat(user_input, hist):
        """
        Creates a chat to be used with gradio chatbot function.
        Every time a message is sent a history list is constructed to give the chatbot context.
        Response takes the user input and history to invoke our chain.
        """
        langchain_history = []
        for item in hist:
            if item["role"] == "user":
                langchain_history.append(HumanMessage(content=item["content"]))
            elif item["role"] == "history":
                langchain_history.append(AIMessage(content=item["content"]))

        response = chain.invoke({"input": user_input, "history": langchain_history})

        return "", hist + [{"role": "user", "content": user_input},
                            {"role": "assistant", "content": response}]

    def clear_chat():
        """
        Returns an empty string for msg and an empty list for history which will clear the chat.
        """
        return "", []

    # Gradio setup
    page = gr.Blocks(
        title="Chat with Einstein"
    )

    # Add widgets to page.
    with page:
        # Basic title and description
        gr.Markdown(
            """
            # Chat with Einstein
            Welcome to your chat with Einstein
            """
        )

        # Chatbot widget to allow for conversation wit bot
        chatbot = gr.Chatbot(
            avatar_images=[None, "einstein.png"],
            show_label=False
        )

        # Textbox widget for user input
        msg = gr.Textbox(show_label=False, placeholder="Ask Einstein!")

        # When a message is submitted our chat function is called sending the msg and chatbot variables.
        msg.submit(chat, [msg, chatbot], [msg, chatbot])

        # Clear Button which clears outputs and history.
        clear = gr.Button("Clear Chat", variant="secondary")
        clear.click(clear_chat, outputs=[msg, chatbot])

        page.launch(share=True)

if __name__ == "__main__":
    main()