import os

# Start the local inference server using Docker
os.system("pip install inference-cli")
os.system("inference server start")
