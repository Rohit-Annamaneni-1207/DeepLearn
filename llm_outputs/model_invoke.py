# import langgraph
# from langchain_community.chat_models import ChatOllama
# import ollama
from typing import TypedDict, List, Dict
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from ollama import chat, ChatResponse
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# MODEL_NAME = "qwen2.5:1.5b"
MODEL_NAME = "phi3:mini"
# llm = ChatOllama(model = MODEL_NAME, temperature = 0.7)


class Concept(BaseModel):
    concept: str = Field(description="The concept")
    definition: str = Field(description="The definition of the concept")
    # importance: float = Field(description="The importance of the concept")

class Concepts(BaseModel):
    concepts: List[Concept] = Field(description="The concepts")

class Node(BaseModel):
    node: str = Field(description="The node")
    description: str = Field(description="The description of the node")
    children: List['Node'] = Field(description="The children of the node")

class RootNode(BaseModel):
    root: str = Field(description="The root of the mindmap")
    description: str = Field(description="The description of the root")
    children: List[Node] = Field(description="The children of the mindmap")

class Question_Answer(BaseModel):
    question: str = Field(description="The question")
    answer: str = Field(description="The answer")

class Quiz(BaseModel):
    quiz: List[Question_Answer] = Field(description="The quiz")

# llm_concept = ChatOllama(model = MODEL_NAME, temperature = 0.7).with_structured_output(Concepts)
# llm_mindmap = ChatOllama(model = MODEL_NAME, temperature = 0.7).with_structured_output(RootNode)

format_concept_extraction_example = """
{
    "concepts": [
        {
            "concept": "conept 1 (Text)",
            "definition": "definition 1 (Text)"
        },
        {
            "concept": "concept 2 (Text)",
            "definition": "definition 2 (Text)"
        },
        {
            "concept": "concept 3 (Text)",
            "definition": "definition 3 (Text)"
        }]
}
"""


format_mindmap_example = """
{
  "root": "concept 1 (str)",
  "description": "description 1 (str)",
  "children": [
    {
      "node": "concept 2 (str)",
      "description": "description 2 (str)",
      "children": [
        {"node": "concept 3 (str)", "description": "description 3 (str)", "children": [
            {"node": "concept 4 (str)", "description": "description 4 (str)", "children": []},
            {"node": "concept 5 (str)", "description": "description 5 (str)", "children": []}
        ]},
        {"node": "concept 6 (str)", "description": "description 6 (str)", "children": []},
        {"node": "concept 7 (str)", "description": "description 7 (str)", "children": []}
      ]
    },
    {
      "node": "concept 8 (str)",
      "description": "description 8 (str)",
      "children": [
        {"node": "concept 9 (str)", "description": "description 9 (str)", "children": []},
        {"node": "concept 10 (str)", "description": "description 10 (str)", "children": []},
        {"node": "concept 11 (str)", "description": "description 11 (str)", "children": []}
      ]
    }
  ]
}
"""

def model_invoke_summary(retrieved_docs):
    system_prompt = f"""You are a helpful assistant that summarizes long documents. Read the given text, and summarize accurately. Do not hallucinate. Do not give inconsistent summaries. Keep the summary concise and reflective of the given text. Give ONLY the summary. Here are retrieved chunks from the document: 
    {" ".join([doc.page_content for doc in retrieved_docs])}"""

    options = {'temperature': 0.7}
    response = chat(model=MODEL_NAME, messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ], options = options)
    return response.message.content

def model_invoke_qna(query, retrieved_docs):
    system_prompt = f"""You are a helpful assistant that answers questions based on the given text. Read the given text, and answer the question accurately. Do not hallucinate. Do not give inconsistent answers. Give ONLY the answer. Here are retrieved chunks from the document: 
    {" ".join([doc.page_content for doc in retrieved_docs])}"""

    options = {'temperature': 0.7}
    response = chat(model=MODEL_NAME, messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": query
        }
    ], options = options)
    return response.message.content

def model_invoke_generate_quiz(topic, retrieved_docs):
    system_prompt = f"""You are a helpful assistant that generates a quiz based on the given text. Read the given text, and generate a quiz accurately. Do not hallucinate. Make sure the quiz questions are related to the topic. Give ONLY the quiz. The quiz topic is: {topic}. Here are retrieved chunks from the document: 
    {" ".join([doc.page_content for doc in retrieved_docs])}\n\n"""

    options = {'temperature': 0.7}
    response = chat(model=MODEL_NAME, messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ], options = options, format=Quiz.model_json_schema())
    response = response.message.content
    try:
        response = Quiz.parse_raw(response)
    except:
        response = None
    return response

def concept_extraction(retrieved_docs, format_example = format_concept_extraction_example):
    system_prompt = f"""You are a helpful assistant that extracts concepts from the given text. Read the given text, and extract concepts ACCURATELY. Extract AS MANY topics as possible. Do not hallucinate. Do not give inconsistent concepts. Here are retrieved chunks from the document: 
    {" ".join([doc.page_content for doc in retrieved_docs])}\n\n

    Rules:
    - Use plain ASCII characters only
    - Do NOT use backslashes (\\)
    - Do NOT use LaTeX or math symbols
    - Do NOT escape characters
    - Write definitions in plain English"""

    # Only output in the given format. Do NOT give any other text other than the provided format:\n {format_example}"""
    options = {'temperature': 0.8}
    response = chat(model=MODEL_NAME, messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ], options = options, format=Concepts.model_json_schema(),)

    response = response.message.content
    # response = """{"concepts": """ + response + "}"
    # validate, dont return
    
    print("CONCEPTS RAW ===========================================================================")
    print(response)
    print("=======================================================================================")
    
    try:
        response = Concepts.parse_raw(response)
    except:
        return None
    return response

def generate_mindmap(concepts: Concepts, format_example = format_mindmap_example):
    # only output in the given format to be read by pydantic model

    concept_string = "\n".join(["{\"concept\": \"" + concept.concept + "\", \"definition\": \"" + concept.definition + "\"}" for concept in concepts.concepts])
    system_prompt = f"""You are a helpful assistant that generates mindmaps based on the given text. Read the given text, and generate a mindmap accurately in the form of a TREE.
    - The concepts are given in curly brackets
    - ONLY REARRANGE THIS CONCEPTS INTO A TREE, DO NOT CREATE NEW CONCEPTS
    - root/node is the concept name
    - description is the concept definition
    - children is the list of child concepts
    - Do NOT hallucinate
    - Do NOT give inconsistent mindmaps
    - Do NOT create any concepts not provided
    
    Here are retrieved chunks from the document: 
    {concept_string}"""

    options = {'temperature': 0.8}
    response = chat(model=MODEL_NAME, messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ], options = options, format=RootNode.model_json_schema())

    response = response.message.content

    print("MINDMAP RAW ===========================================================================")
    print(response)
    print("=======================================================================================")
    # validate, dont return
    
    try:
        response = RootNode.parse_raw(response)
    except:
        return None
    return response


if __name__ == "__main__":

    R = RootNode(root="root", description="description", children=[])
    N = Node(node="node", description="description", children=[])
    C = Concept(concept="concept", definition="definition")
    print(type(R))
    print(type(N))
    print(type(C))


    # retrieved_docs = [Document(page_content="This is a test document.")]
    # print(model_invoke_summary(retrieved_docs))