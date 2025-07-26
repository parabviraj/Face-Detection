from fastapi import FastAPI, File, UploadFile, HTTPException
from deepface import DeepFace
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

DETECTOR_PRIORITY = ["opencv", "retinaface", "mediapipe", "mtcnn"]

@app.get("/")
async def root():
    return {"message": "Welcome to the improved DeepFace API!"}

async def read_image_async(file: UploadFile) -> np.ndarray:
    try:
        contents = await file.read()
        if not contents:
            raise ValueError("Empty file uploaded.")

        # First pass check for corruption
        image = Image.open(io.BytesIO(contents))
        image.verify()

        # Reload after verification
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return np.array(image)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception as e:
        logger.error(f"Image read failed: {e}")
        raise HTTPException(status_code=400, detail="Unable to process image. Ensure it is a valid JPG/PNG.")

def get_embedding_with_fallback(img: np.ndarray):
    for backend in DETECTOR_PRIORITY:
        try:
            result = DeepFace.represent(
                img_path=img,
                model_name="Facenet",
                detector_backend=backend,
                enforce_detection=True
            )
            logger.info(f"Face detected using backend: {backend}")
            return result, backend
        except Exception as e:
            logger.warning(f"{backend} failed: {e}")
    raise HTTPException(status_code=400, detail="Face not detected using any backend.")

@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, or PNG images are allowed.")
    
    image_np = await read_image_async(file)
    embeddings, backend_used = get_embedding_with_fallback(image_np)

    embedding = embeddings[0].get("embedding")
    if embedding is None:
        raise HTTPException(status_code=400, detail="Face not detected.")

    return {
        "embedding": embedding,
        "detector_backend": backend_used,
        "model": embeddings[0].get("model")
    }

@app.post("/verify")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    for f in [file1, file2]:
        if not f.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Only JPG, JPEG, or PNG images are allowed.")
    
    img1 = await read_image_async(file1)
    img2 = await read_image_async(file2)

    for backend in DETECTOR_PRIORITY:
        try:
            result = DeepFace.verify(
                img1_path=img1,
                img2_path=img2,
                model_name="Facenet",
                detector_backend=backend,
                enforce_detection=True
            )
            logger.info(f"Verification successful using backend: {backend}")
            return {
                "verified": result.get("verified"),
                "distance": result.get("distance"),
                "threshold": result.get("threshold"),
                "model": result.get("model"),
                "similarity_metric": result.get("similarity_metric"),
                "detector_backend": backend
            }
        except Exception as e:
            logger.warning(f"Verification failed using {backend}: {e}")

    raise HTTPException(status_code=400, detail="Face verification failed using all available backends.")
