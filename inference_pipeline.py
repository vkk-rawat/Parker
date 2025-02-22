# Import the InferencePipeline object
from inference import InferencePipeline
import cv2

def my_sink(result, video_frame):
    if result.get("output_image"): # Display an image from the workflow response
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)
    print(result) # do something with the predictions of each frame
    

# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="vMt0JUmSrXKPCmTa9jOO",
    workspace_name="shiv-vihar",
    workflow_id="detect-count-and-visualize",
    video_reference=r"C:\Users\Vivek Rawat\OneDrive\Desktop\WhatsApp Video 2025-02-22 at 07.07.51_efa8e6cd.mp4"

, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=100,
    on_prediction=my_sink
)
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish
