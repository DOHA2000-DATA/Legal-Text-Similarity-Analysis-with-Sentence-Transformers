# Legal-Text-Similarity-Analysis-with-Sentence-Transformers

##  Project Description

This project focuses on analyzing and comparing legal texts using Machine Learning (ML) and Natural Language Processing (NLP) techniques. 

The primary objective is to automatically measure the similarity between legal questions and legal articles to improve information retrieval, legal document classification,

and automated legal assistance.

The project uses Sentence Transformers, a model that converts textual data into high-dimensional vector representations, allowing for similarity computation. This approach can be useful for:

    Legal document search: Finding the most relevant laws or regulations related to a given legal question.

    Automated legal assistance: Suggesting relevant case laws or regulations based on user queries.
    
    Legal text classification: Grouping similar legal texts for further analysis.

🚀 Technologies Used

The project leverages various tools and frameworks to perform legal text similarity analysis:
Technology	Usage
Python	Main programming language

Sentence Transformers	Converts text into vector embeddings

Scikit-learn	Model evaluation and similarity metrics

Pandas	Data manipulation and preprocessing

NumPy	Numerical operations

Matplotlib / Seaborn	Data visualization

Google Colab / Jupyter Notebook	Model training and experimentation

Google Drive	Storing datasets


📂 Project Structure

The project is organized as follows:

📁 legal_similarity_analysis/

│── 📂 data/               # Raw and processed legal text datasets

│── 📂 models/             # Pretrained and fine-tuned models

│── 📂 notebooks/          # Jupyter notebooks for training and testing

│── 📂 results/            # Similarity scores and evaluations

│── preprocess.py         # Data preprocessing script

│── train.py              # Model training script

│── evaluate.py           # Model evaluation and similarity analysis

│── requirements.txt      # List of dependencies

│── README.md             # Project documentation

🔧 Installation and Setup

To run the project, follow these steps:

1️⃣ Clone the Repository

git clone https://github.com/your-repo/legal_similarity_analysis.git
cd legal_similarity_analysis

2️⃣ Create a Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

If you're running the project on Google Colab, install missing dependencies:

!pip install transformers datasets scikit-learn pandas sentence-transformers

# Model Training

To train the Sentence Transformer model on legal text data, run:

python train.py --epochs 10 --batch-size 16 --learning-rate 0.0001

    --epochs: Number of training iterations
    
    --batch-size: Number of samples per training step
    
    --learning-rate: Model learning rate

Alternatively, use Google Colab to train with GPU acceleration.

# Evaluating the Model

Once trained, the model can be evaluated on a test dataset to measure its performance:

python evaluate.py

This script will:

    Compute similarity scores between legal questions and articles.
    
    Use metrics such as cosine similarity, Euclidean distance, and accuracy.
    
    Generate visualizations of similarity distributions.

🌍 Using the Model

After training, the model can be used to find the most relevant legal articles for a given question.

Example usage in Python:

from sentence_transformers import SentenceTransformer

# Load the trained model

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Example query

query = "What are the laws on digital privacy?"

# Encode the query into a vector

query_embedding = model.encode(query)

print("Query embedding shape:", query_embedding.shape)

Finding the Most Similar Legal Articles

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example legal articles

legal_articles = [
    "The Data Protection Act ensures user privacy.",
    "Privacy laws regulate the collection of personal data.",
    "Intellectual property laws protect creative works."
]

# Encode articles
article_embeddings = model.encode(legal_articles)

# Compute cosine similarity
similarities = cosine_similarity([query_embedding], article_embeddings)

# Display most relevant article
most_similar_index = np.argmax(similarities)
print("Most relevant legal article:", legal_articles[most_similar_index])

#  Results & Visualization

To analyze model performance, similarity distributions and clustering methods can be used. The project includes visualizations such as:

✅ Heatmaps of similarity scores

✅ t-SNE or PCA projections of embeddings

✅ Performance metrics on test data

Example:

--import seaborn as sns

--import matplotlib.pyplot as plt

--sns.heatmap(similarities, annot=True, cmap="Blues")

--plt.title("Legal Text Similarity Heatmap")

--plt.show()




