import os
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
from imagebind.models import imagebind_model
from dotenv import load_dotenv


load_dotenv()

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))
img_path = os.getenv("IMG_PATH")
img_dirs = os.listdir(img_path) 
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data_into_qdrant(collection_name):
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection( 
        collection_name = collection_name, 
        vectors_config = { 
            "image": VectorParams( size = 1024, distance = Distance.COSINE ),
            # TODO: Add audio and video vector functionality
            #"audio": VectorParams( size = 1024, distance = Distance.COSINE ), 
            #"video": VectorParams( size = 1024, distance = Distance.COSINE )
        } 
    )
        
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    # Load the data
    media_objs = list()
    for idx, name in enumerate(img_dirs):
        if name.endswith(".png") or name.endswith(".jpg"):
            print(f"Adding {name}")
            path = img_path + name

            inputs = {
                ModalityType.VISION: data.load_and_transform_vision_data([path], device),
            }
            
            with torch.no_grad():
                embeddings = model(inputs)
            
            media_objs.append(PointStruct(
                id=idx, 
                payload= {
                    "name": name.split(".")[0], 
                    "path": path,              
                },
                vector={"image": embeddings["vision"].flatten().tolist(),} 
                        # TODO: Add audio and video vector functionality
                        #"audio": embeddings["audio"].flatten().tolist(), 
                        #"video": embeddings["video"].flatten().tolist(),}
            ))

    qdrant_client.upsert(collection_name=collection_name, points=media_objs)

if __name__ == "__main__":
    load_data_into_qdrant("demo-collection")