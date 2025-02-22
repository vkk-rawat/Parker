
from inference import InferencePipeline
import cv2

def my_sink(result, video_frame):
    if result.get("output_image"):  
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)
    print(result)  # 


pipeline = InferencePipeline.init_with_model(
    model_id="Vivek Rawat/detect-count-and-visualize",
    api_key="vMt0JUmSrXKPCmTa9jOO",
    workspace_name="shiv-vihar",
    video_reference=0, 
    max_fps=30,
    on_prediction=my_sink
)

pipeline.start()  
pipeline.join()  
