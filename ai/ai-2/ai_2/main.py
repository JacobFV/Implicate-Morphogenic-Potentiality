import os
from langchain.agents import Tool, AgentExecutor, ConversationalChatAgent
from langchain.memory import PostgresChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_experimental.utilities import PythonREPL

# Set up PostgreSQL connection
POSTGRES_URI = os.environ.get(
    "POSTGRES_URI", "postgresql://user:password@localhost/dbname"
)

# Create PostgreSQL memory
message_history = PostgresChatMessageHistory(
    connection_string=POSTGRES_URI,
    session_id="default_session",  # You can change this for different sessions
)

# Create ConversationBufferMemory with PostgresChatMessageHistory
memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=message_history, return_messages=True
)  # ... (previous code)

# Create Python REPL tool
python_repl = PythonREPL()

tools = [
    Tool(
        name="Python REPL",
        func=python_repl.run,
        description="A Python REPL. Use this to execute Python commands. Input should be a valid Python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    )
]  # ... (previous code)

# Set up the language model
llm = ChatOpenAI(temperature=0)

# Create the agent
agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True, max_iterations=2
)  # ... (previous code)


def main():
    print("Welcome to the Langchain Conversation Agent!")
    print("You can chat with the agent or ask it to execute Python code.")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        response = agent_executor.run(user_input)
        print("Agent:", response)


if __name__ == "__main__":
    main()
