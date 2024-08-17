import os
import torch
import ollama
import gradio as gr 
from qdrant_client.http.models import NamedVector
from dotenv import load_dotenv
from imagebind import data
from imagebind.models import imagebind_model
from qdrant_client import QdrantClient
import qdrant_client.http.models as models

load_dotenv()

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)

def process_query_ollama(text_query, collection_name="demo-collection"):
    user_input = {"text": data.load_and_transform_text([text_query], device)}
        
    with torch.no_grad():
        user_embeddings = model(user_input)
        # Extract the vision embedding, which is aligned with other modalities
        
    search_params = models.SearchParams(hnsw_ef=128, exact=False)

    hits = qdrant_client.search(
        collection_name=collection_name,
        query_vector=NamedVector(
                name="image",
                vector=user_embeddings['text'][0].tolist()
            ),
        search_params=search_params,
        limit=1
    )

    hit_path = hits[0].payload['path']
    prompt_template = f"Respond to this prompt, using only the provided image as refernce: {text_query}"

    try:
        output = ollama.generate(
          model = "bakllava", # You can also use any other multimodal LLM model here
          prompt = prompt_template,
          images = [hit_path],
        )
    except Exception as e:
        return (f"An error occurred : {e}", None)
    
    return (output['response'], hit_path)


iface = gr.Interface(
    title="Multi-Modal RAG",
    description="by Malte Barth",
    fn=process_query_ollama,
    inputs=[
        gr.Textbox(label="text_query"),
        ],
        # TODO: Add audio and image query functionality
        #gr.Audio(sources="upload", type="filepath"),
        #gr.Image(label="image_query", type="filepath")
    #],
    outputs=[gr.Textbox(label="Text"),
             gr.Image(label="Image")], 
            # TODO: Add audio retrieval functionality
            #gr.Audio(label="Audio"),
)

iface.launch()