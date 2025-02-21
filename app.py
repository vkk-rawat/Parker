from roboflow import Roboflow

rf = Roboflow(api_key="vMt0JUmSrXKPCmTa9jOO")
workspace = rf.workspace("shiv-vihar")
project = workspace.project("detect-count-and-visualize")

print("Model Name:", project.model_id)  # This prints your model name
