# Creative RAG Assistant
The idea of this project is to help a creative project team by storing it's documentation files in a vector database which is used in a RAG approach to power a useful team assistant chatbot based on an LLM.
Creative teams often accumulate mdeia-rich content like images, videos, audio files, and text documents. This project aims to help these teams to store and retrieve their files in a multimodal way.

## Pre-requisites 
1. Create vector database
   
    a. Local setup: [Qdrant](https://qdrant.tech/documentation/quickstart/), [Weaviate](https://weaviate.io/developers/weaviate/installation/docker-compose#starter-docker-compose-file), ...

    b. Cloud setup: using API keys from hosted vector database
2. Install requirements of a multimodal embedding model (e.g. [ImageBind](https://imagebind.metademolab.com/))
3. Setup LLM
   
   a. Local setup: [Ollama](https://github.com/ollama/ollama) -> `ollama run llava` (or any other local multimodal model, like Bakllava)

   b. Cloud setup: OpenAI API, Claude API, ...

## How to use
1. Store files in the vector database
   
   a. Put files into folder and add that folder to .env file

   b. Run `python load_data_into_qdrant.py`
2. Run RAG application
   
   `python rag_pipeline.py`