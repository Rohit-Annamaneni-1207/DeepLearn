import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.model_invoke import model_invoke_summary, model_invoke_qna, concept_extraction, generate_mindmap
from RAG.rag_utils import load_chunk_pdfs, retrieve_from_index, retrieve_all_from_index
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = load_chunk_pdfs(embedding_model = embedding_model)

# retrieved_docs = retrieve_all_from_index(index)

topic = "Linear Regression"
retrieved_docs = retrieve_from_index(index, topic, k = 30)
# summary = model_invoke_summary(retrieved_docs)
concepts = concept_extraction(retrieved_docs)

print(concepts)
mindmap = generate_mindmap(concepts)
print(mindmap)


"""
{"concepts": [
    {
        "concept": "Cross-Validation",
        "definition": "In supervised learning, where we are given a labeled dataset and want to build the best predictive model by selecting one of many candidate models. A common strategy employed is cross validation (CV), which involves dividing data into two segments: training set \(X\) and test/validation set \(\mathbf{T}\). We learn parameters using only the first segment, evaluate performance on both sets with various parameter settings to determine a good setting for that model."
    },
    {
        "concept": "Maximum Mean Discrepancy (MMD)",
        "definition": "An objective function used in statistical learning models. MMD measures how well the predictions of two distributions align, typically between an empirical distribution \( p_1 \) and a prior known probability density, by integrating differences using kernel functions."
    },
    {
        "concept": "Kernel Density Estimation (KDE)",
        "definition": "A non-parametric way to estimate the PDF of continuous random variables. This method uses kernels and bandwidths in a manner similar to Gaussian Processes, but instead utilizes inner products between feature vectors."
    },
    {
        "concept": "Softmax regression",
        "definition": "A generalization for classification problems where the conditional distribution of y given x is not binary. Specifically designed with Assumption3: p(y = i|x; θ) = eηi /Σk_j exp(θT jx)."
    },
    {
        "concept": "Naive Bayes",
        "definition": "A classification algorithm that assumes conditional independence between features, calculates the probability of each class given a set of feature vectors and picks whichereach higher posterior."
    },
    {
        "concept": "Bias Variance Trade-off in Learning Algorithms",
        "definition": "'bias' refers to an underfitting problem where our model does not learn enough, resulting in poor test set performance. 'variance' refers to when the training error is high and learning from noise or outliers."
    },
    {
        "concept": "Least Squares Regression",
        "definition": "'Learning’ can also refer to linear regression, where our objective function J(θ) becomes (XtX)−1 X t y. In this context we find the value of θ that minimizes it by setting its gradients equal."
    },
    {
        "concept": "Support Vector Machine Regression",
        "definition": "'SVR’ is a regression model where our objective function J(w,b) becomes (y - X w )T K y + λ/2 m||Kkx − K k b ||²."
    }
]}
"""
