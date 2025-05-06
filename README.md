# 📚 Book Recommender System

This project is a **Book Recommendation System** that suggests books based on user preferences and book metadata. It uses **content-based filtering**, **vector search**, **text classification**, and **sentiment analysis** techniques to generate intelligent recommendations. The system is built with Python and deployed using **Gradio** for a clean and interactive web interface.

## 🔍 Features

- 🔎 Content-based book recommendation
- 🧠 Vector similarity search using book metadata (e.g., title, description)
- 🏷️ Text classification of book content
- 💬 Sentiment analysis on book descriptions or user reviews (if available)
- ⚡ Fast and interactive deployment using Gradio
- 🗂️ Automatic dataset download from Kaggle via `kagglehub`

## 📦 Dataset

The dataset is automatically downloaded using the `kagglehub` library:

```python
from kagglehub import dataset_download
path = dataset_download("dylanjcastillo/7k-books-with-metadata")
```

This dataset includes metadata for over 7,000 books, such as title, author, genre, and description.

🚀 Getting Started

1. Clone the Repository
   
git clone https://github.com/elifkeskin/book-recommender.git

cd book-recommender

2. (Optional) Create a Virtual Environment
   
python -m venv venv

source venv/bin/activate 

# On Windows: venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Run the Gradio App
   
python gradio-dashboard.py

Gradio will generate a local and public link where you can interact with the app.

🧠 Technologies Used
Python

Pandas, NumPy

Scikit-learn

Gradio (for web UI)

Kagglehub (for dataset access)

Embedding models (for vector search)

HuggingFace Transformers (for sentiment analysis & classification)

📂 Project Structure
book-recommender/
│
├── gradio-dashboard.py             # Gradio application

├── data-exploration.ipynb          # Data exploration

├── vector-search.ipynb             # Similarity-search

├── text-classification.ipynb       # Classifying Book Descriptions (Zero-Shot Classification)

├── sentiment-analysis.ipynb        # Sentiment analysis for all book descriptions

└── README.md                       # This page

🎯 Functionality
retrieve_semantic_recommendations(): Finds similar books using  by similarity_-search

generate_predictions(): Returns the category with the maximum score probability.

calculate_max_emotion_scores(): Keeps all scores for a single statement.It reveals the score point for each emotion and adds it using the correct tag.

📸 Sample UI

<img width="604" alt="Semantic Book Recommender" src="https://github.com/user-attachments/assets/90e3051a-5cae-490c-a543-0019f0feec28" />

<img width="604" alt="Book Recomender" src="https://github.com/user-attachments/assets/3d2daa01-ad3f-4f67-b342-f89ecdc18cdc" />

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.



