# Base image with Python 3.9 and CUDA
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Prevent interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies, including libglib2.0-0 and other OpenCV dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies inside the yolov10 folder
COPY yolov10/ /app/yolov10/
WORKDIR /app/yolov10
RUN pip install .

# Copy the entire project directory into the container
WORKDIR /app
COPY . .

# Set the command to run app.py with command-line arguments for input and output files
CMD ["python", "app.py", "input_file.json", "output_file.json"]
