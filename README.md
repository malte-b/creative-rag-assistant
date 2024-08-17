# Creative RAG Assistant
The idea of this project is to help a creative project team by storing it's documentation files in a vector database which is used in a RAG approach to power a useful team assistant chatbot based on an LLM.
Creative teams often accumulate mdeia-rich content like images, videos, audio files, and text documents. This project aims to help these teams to store and retrieve their files in a multimodal way.

## How to use   
1. Create vector database
    a. Local setup: [Qdrant](https://qdrant.tech/documentation/quickstart/), [Weaviate](https://weaviate.io/developers/weaviate/installation/docker-compose#starter-docker-compose-file), ...
    b. Cloud setup: using API keys from hosted vector database
2. Create a collection for your files
3. Embed your files with a multimodal embedding model (e.g. [ImageBind](https://imagebind.metademolab.com/))
4. Add your embedded files to your collection
5. Prompt an LLM about your files
   a. Local setup: [Ollama](https://github.com/ollama/ollama) -> `ollama run llava` (or any other local multimodal model, like Bakllava)
   b. Cloud setup: OpenAI API, Claude API, ...