#dependecies
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import  OpenAIEmbeddings
from langchain_text_splitters import  CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
from pyparsing import results

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800" #much better resolution
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.png",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

# filtering based on category and sorting based on emotional tone
def retrieve_semantic_recommendations(
        query:str,
        category : str = None,
        tone:str = None,
        initial_top_k :int =50,
        final_top_k: int = 16,
) -> pd.DataFrame:
     # getting book recomendations from vector database (db_base) limiting by initial_top_k
     recs = db_books.similarity_search(query, k=initial_top_k)
     # getting isbns of those recommendations by splitting them off the descriptions
     books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
     # limiting books df to just those that match the isbns of those book's.
     book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

     # We have drop down menu on our dashboard.It can either read all or it can read one of the four simple categories.
     if category != "All":
         book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
     else:
         book_recs = book_recs.head(final_top_k)

    # sorting those recommendations based on the highest probabilities
     if tone == "Happy":
         book_recs.sort_values(by="joy", ascending=False, inplace=True)
     elif tone == "Surprising":
         book_recs.sort_values(by="surprise", ascending=False, inplace=True)
     elif tone == "Angry":
         book_recs.sort_values(by="anger", ascending=False, inplace=True)
     elif tone == "Suspenseful":
         book_recs.sort_values(by="fear", ascending=False, inplace=True)
     elif tone == "Sad":
         book_recs.sort_values(by="sadness", ascending=False, inplace=True)

     return book_recs

# specifies what we want to display on gradio dashboard. This is called "recommend_books" functions.
def recommend_books(
         query:str,
         category : str ,
         tone:str ,

 ):
     recommendations = retrieve_semantic_recommendations(query, category, tone)
     results = []

     # loop over every single one of these recommendations
     for _, row  in recommendations.iterrows():
         # Our dashboard has limited space so, we don't necessarily want to show full description
         description = row["description"]
         truncated_desc_split = description.split()
         # if the description has more than 20 words what we're going to do is cut it off and just make it continue with a trailing ellipses.
         truncated_description = " ".join(truncated_desc_split[:20]) + "..."

         # In this dataset, if a book has more than one author, they are combined using a semicolon.

         authors_split = row["authors"].split(";")
         if len(authors_split) == 2:
             authors_str = f"{authors_split[0]} and {authors_split[1]}"
         elif len(authors_split) > 2:
              authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
         else:
             authors_str = row["authors"]

         caption = f"{row['title']} by {authors_str}:{truncated_description}"
         results.append((row["large_thumbnail"], caption))

     return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g, A story about forgiveness.")
        category_dropdown = gr.Dropdown(choices = categories, label="Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label="Select a emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns=8, rows=2)

    submit_button.click(fn = recommend_books,
                            inputs = [user_query, category_dropdown, tone_dropdown],
                            outputs = output)

    if __name__ == "__main__":
         dashboard.launch(share=True)

      
