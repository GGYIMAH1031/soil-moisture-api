# Use python:3.10-slim as the base image
FROM python:3.10-slim

# Set /app as the working directory
WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt .

# Install basic Python dependencies
RUN pip install --upgrade pip setuptools wheel --no-cache-dir

# Install PyTorch and torchvision separately to ensure the correct versions and configurations
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install autogluon separately (if it has specific version requirements, consider adding it to the requirements.txt)
RUN pip install autogluon --no-cache-dir

# Install the rest of the requirements
RUN pip install -r requirements.txt --no-cache-dir

# Copy the rest of the application's source code
COPY . .

# Specify the default command for the container
CMD ["python", "app.py"]


