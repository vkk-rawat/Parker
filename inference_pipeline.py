from inference import InferencePipeline
import cv2

# Function to process predictions
def my_sink(result, video_frame):
    if result.get("output_image"):
        cv2.imshow("Inference Output", result["output_image"].numpy_image)
        cv2.waitKey(1)
    print("Predictions:", result.get("predictions", []))

pipeline = InferencePipeline.init_with_model(
    api_key="vMt0JUmSrXKPCmTa9jOO",
    model_id="detect-count-and-visualize",  # Replace with actual model ID
    video_reference=0,
    max_fps=30,
    on_prediction=my_sink
)

# Start inference
pipeline.start()
pipeline.join()
