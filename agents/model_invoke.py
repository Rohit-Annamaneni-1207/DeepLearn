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
    children: List['Node'] = Field(description="The children of the node")

class RootNode(BaseModel):
    root: str = Field(description="The root of the mindmap")
    children: List[Node] = Field(description="The children of the mindmap")

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
    # Here is an example:
    # [
    #     {
    #         "concept": "supervised learning",
    #         "definition": "The process of learning from labeled data to make predictions."
    #     },
    #     {
    #         "concept": "Regression",
    #         "definition": "A type of supervised learning where the target variable is continuous."
    #     },
    #     {
    #         "concept": "Linear regression",
    #         "definition": "In linear regression, a model that predicts a dependent variable based on one or more independent variables using the equation hθ(x) = θ0 + θ1x1 + ... + θnxn."
    #     }
    # ]
    # """

format_mindmap_example = """
{
  "root": "Neural Network Training",
  "children": [
    {
      "node": "Optimization",
      "children": [
        {"node": "Gradient Descent", "children": []},
        {"node": "Momentum", "children": []},
        {"node": "Adam", "children": []}
      ]
    },
    {
      "node": "Regularization",
      "children": [
        {"node": "L1", "children": []},
        {"node": "L2", "children": []},
        {"node": "Dropout", "children": []}
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

    print(response)
    
    try:
        response = Concepts.parse_raw(response)
    except:
        return None
    return response

def generate_mindmap(concepts: Concepts, format_example = format_mindmap_example):
    # only output in the given format to be read by pydantic model

    concept_string = "\n".join(["{\"concept\": \"" + concept.concept + "\", \"definition\": \"" + concept.definition + "\"}" for concept in concepts.concepts])
    system_prompt = f"""You are a helpful assistant that generates mindmaps based on the given text. Read the given text, and generate a mindmap accurately. Do not hallucinate. Do not give inconsistent mindmaps. Here are retrieved chunks from the document: 
    {concept_string}\n\n
    Only output in the given example format:\n {format_example}"""

    options = {'temperature': 0.8}
    response = chat(model=MODEL_NAME, messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ], options = options)

    response = response.message.content
    # validate, dont return
    
    # try:
    #     RootNode.parse_raw(response)
    # except:
    #     return None
    return response


if __name__ == "__main__":

    resp = """
    {
    "concepts": [
        {
            "concept": "bias-variance tradeoff",
            "definition": "Bias and variance are two fundamental characteristics of machine learning models. High bias can cause a model to be overly simplistic, leading to underfitting where the model is too rigid to capture underlying trends in data (sometimes called 'blindness'). Conversely, high variance might lead the model to adapt too much to specific features or noise present only in training datasets - this can make it perform poorly on unseen testing samples. It's a tradeoff that demands balancing between bias and reducing its counterpart."
        },
        {
            "concept": "softmax regression",
            "definition": "Softmax function is an extension of logistic regression to multi-class classification problems where the output can take more than two classes. In softmax regression, it models a categorical distribution whose elements sum up to 1 and represent probabilities that define class membership for each input x."
        },
        {
            "concept": "kernel trick",
            "definition": "The kernel function is pivotal in support vector machines (SVMs). Instead of directly computing the inner product between two inputs, which can be computationally expensive and complex if data isn't linearly separable, SVM applies a non-linear transformation using kernels. This method allows algorithms to operate within higher dimensions implicitly."     
        },
    ]
}
    """

    print(Concepts.parse_raw(resp))


    # retrieved_docs = [Document(page_content="This is a test document.")]
    # print(model_invoke_summary(retrieved_docs))