import sys
import os
import streamlit as st
import streamlit_agraph as stgraph
from streamlit_agraph import Node as stgraphNode
from streamlit_agraph import Edge as stgraphEdge
from streamlit_agraph import Config, agraph
from langchain_community.embeddings import HuggingFaceEmbeddings

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_outputs import model_invoke
# from llm_outputs.model_invoke import Concept, Concepts, Node, RootNode, Question_Answer, Quiz
from RAG.rag_utils import load_chunk_pdfs, retrieve_from_index, retrieve_all_from_index

# Setup page config
st.set_page_config(layout="wide", page_title="DeepLearn")

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# Initialize session state variables
if 'index' not in st.session_state:
    st.session_state.index = None
if 'docs_processed' not in st.session_state:
    st.session_state.docs_processed = False
if 'summary_result' not in st.session_state:
    st.session_state.summary_result = ""
if 'mindmap_data' not in st.session_state:
    st.session_state.mindmap_data = None
if 'mindmap_desc_map' not in st.session_state:
    st.session_state.mindmap_desc_map = {}
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None

def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        # Clear docs folder
        docs_path = os.path.abspath("docs")
        if not os.path.exists(docs_path):
            os.makedirs(docs_path)

        for file in os.listdir(docs_path):
            os.remove(os.path.join(docs_path, file))

        file_path = os.path.join(docs_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Processing document..."):
            index = load_chunk_pdfs(embedding_model=embedding_model)
            st.session_state.index = index
            st.session_state.docs_processed = True
            
            # Reset results when new file is processed
            st.session_state.summary_result = ""
            st.session_state.mindmap_data = None
            st.session_state.mindmap_desc_map = {}
            st.session_state.quiz_data = None
            
        st.success("File uploaded and processed successfully")

if __name__ == "__main__":
    st.title("DeepLearn")
    st.subheader("RAG Powered Learning Assistant")

    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        
        # logic to process file
        if uploaded_file:
            if st.button("Process PDF"):
                process_uploaded_file(uploaded_file)
        
        if st.session_state.index:
            st.success("Index Ready")

    if st.session_state.index:
        tabs = st.tabs(["Summary", "QnA", "Quiz", "Mindmap"])
        
        # --- Summary Tab ---
        with tabs[0]:
            st.title("Summary")
            topic = st.text_input("Enter your topic for summary", key="summary_topic")
            if st.button("Generate Summary"):
                if topic:
                    with st.spinner("Generating summary..."):
                        docs = retrieve_from_index(st.session_state.index, topic)
                        summary = model_invoke.model_invoke_summary(docs)
                        st.session_state.summary_result = summary
                else:
                    st.warning("Please enter a topic.")
            
            if st.session_state.summary_result:
                st.write(st.session_state.summary_result)
        
        # --- QnA Tab ---
        with tabs[1]:
            st.title("QnA")
            query = st.text_input("Enter your query", key="qna_query")
            if st.button("Get Answer"):
                if query:
                    with st.spinner("Fetching answer..."):
                        docs = retrieve_from_index(st.session_state.index, query)
                        answer = model_invoke.model_invoke_qna(query, docs)
                        st.write(answer)
                else:
                    st.warning("Please enter a query.")

        # --- Quiz Tab ---
        with tabs[2]:
            st.title("Quiz")
            topic = st.text_input("Enter your topic for quiz", key="quiz_topic")
            if st.button("Generate Quiz"):
                if topic:
                    with st.spinner("Generating quiz..."):
                        docs = retrieve_from_index(st.session_state.index, topic)
                        quiz = model_invoke.model_invoke_generate_quiz(topic, docs)
                        st.session_state.quiz_data = quiz
                else:
                    st.warning("Please enter a topic.")
            
            if st.session_state.quiz_data:
                 for i, question in enumerate(st.session_state.quiz_data.quiz):
                    st.markdown(f"**Q{i+1}: {question.question}**")
                    with st.expander("Reveal Answer"):
                        st.write(question.answer)

        # --- Mindmap Tab ---
        with tabs[3]:
            st.title("Mindmap")
            topic = st.text_input("Enter your topic for mindmap", key="mindmap_topic")
            
            if st.button("Generate Mindmap"):
                if topic:
                    with st.spinner("Generating mindmap..."):
                        docs = retrieve_from_index(st.session_state.index, topic)
                        concepts = model_invoke.concept_extraction(docs)
                        if concepts:
                            mindmap = model_invoke.generate_mindmap(concepts)
                            if mindmap:
                                nodes = []
                                edges = []
                                desc_map = {}

                                def build_graph(node, parent=None, count="0"):
                                    if hasattr(node, 'root'):
                                         label = node.root
                                    elif hasattr(node, 'node'):
                                         label = node.node
                                    else:
                                        label = "Unknown"

                                    desc_map[label] = node.description

                                    nodes.append(
                                        stgraphNode(
                                            id=count,
                                            label=label,
                                            size=25,
                                            shape="dot",
                                            font={'color': 'white'}
                                        )
                                    )
                                    
                                    if parent:
                                        edges.append(stgraphEdge(source=parent, target=count))

                                    child_count = 0
                                    for child in node.children:
                                        build_graph(child, parent=count, count=count + "_" + str(child_count))
                                        child_count += 1

                                build_graph(mindmap)
                                
                                st.session_state.mindmap_data = (nodes, edges)
                                st.session_state.mindmap_desc_map = desc_map
                        else:
                            st.error("Could not extract concepts.")
                else:
                    st.warning("Please enter a topic.")

            if st.session_state.mindmap_data:
                nodes, edges = st.session_state.mindmap_data
                config = Config(
                    width=750,
                    height=600,
                    directed=True,
                    physics=False,
                    hierarchical=True,
                    sortMethod='directed',
                    direction='UD', # Up-Down
                    nodeSpacing=200,
                    levelSeparation=150
                )
                
                selected = agraph(nodes=nodes, edges=edges, config=config)
                
                if selected:
                    # Map unique ID back to label for lookup
                    selected_node = next((n for n in nodes if n.id == selected), None)
                    if selected_node:
                        selected_label = selected_node.label
                        if selected_label in st.session_state.mindmap_desc_map:
                             st.subheader(selected_label)
                             st.write(st.session_state.mindmap_desc_map[selected_label])
    else:
        st.info("Please upload a PDF document and click 'Process PDF' to begin.")