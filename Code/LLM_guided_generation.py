
import openai
import os
import getpass
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

# Ensure OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# Define the structure for the state (Question, Context, Answer)
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

prompt = """
You are an expert in querying graph databases with Cypher. 
Answer the user's question by generating an appropriate Cypher query. 
Use the provided paths to help create the query with unions. 
Ensure that the query considers all relationships and entities in the different paths.

Context, with (entity,1, entity2),Link: {context}

Question: {question}
"""

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.format(question=state["question"], context=docs_content)
    print(messages)
    response = llm.invoke(messages)
    return {"answer": response.content}

def generate_with_paths(query: str, final_paths: List[List[str]]):
    context = str(final_paths)
    state = State(
        question=query,
        context=[Document(page_content=context)],
        answer=""
    )
    print(state)
    return generate(state)



user_query = "Which flights may be consider having risk?"

# Change the model to gpt-4o for the next scenarios
model_name = "gpt-4o-mini"  # Default model for initial runs
llm = init_chat_model("gpt-4o-mini", model_provider="openai")


print("\n🔹 Scenario 1 (gpt4 mini- specific paths): Generate Cypher query based on Paths as context\n")
answer_1 = generate_with_paths(user_query, subschema_section)
print(f"Generated Cypher Query: {answer_1['answer']}\n")


print("\n🔹 Scenario 2(gpt4 mini- all paths): Generate Cypher query based on Paths as context\n")
answer_2 = generate_with_paths(user_query, schema_relations_named)
print(f"Generated Cypher Query: {answer_2['answer']}\n")


# Change the model to gpt-4o for the next scenarios
model_name = "gpt-4o"
llm = init_chat_model("gpt-4o", model_provider="openai")

print("\n🔹 Scenario 1(gpt4- specific paths): Generate Cypher query based on Paths as context\n")
answer_1 = generate_with_paths(user_query, subschema_section)
print(f"Generated Cypher Query: {answer_1['answer']}\n")


print("\n🔹 Scenario 2(gpt4- all paths): Generate Cypher query based on Paths as context\n")
answer_2 = generate_with_paths(user_query, schema_relations_named)
print(f"Generated Cypher Query: {answer_2['answer']}\n")
