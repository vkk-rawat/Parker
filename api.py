from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from inference import InferencePipeline
import uvicorn

app = FastAPI()

# Initialize the inference pipeline with the local server
pipeline = InferencePipeline.init_with_server(
    api_url="http://localhost:9001",
    workspace_name="shiv-vihar",
    workflow_id="detect-count-and-visualize"
)

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run inference
    result = pipeline.infer(image)
    return {"predictions": result.get("predictions", [])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
