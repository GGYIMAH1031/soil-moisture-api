
# Use python:3.10-slim as the base image for a lightweight container
FROM python:3.10-slim

# Set /app as the working directory. All the commands below will be run in this directory.
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
# This prevents re-installing all dependencies upon every build if the requirements didn't change
COPY requirements.txt .

# Install dependencies in a single RUN command to reduce image layers
# Also, combine the pip upgrade and other installations into one layer to reduce the overall size
# We're using `--no-cache-dir` to not store the index on disk, making the image smaller
# Note that if torch and torchvision versions are specified in the requirements.txt,
# this step can be simplified further
RUN pip install -U pip setuptools wheel --no-cache-dir && \
    pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --find-links https://download.pytorch.org/whl/cpu --no-cache-dir && \
    pip install autogluon --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir

# Now copy the rest of the app's source code
COPY . .

# Specify the default command to run when starting the container
CMD ["python", "app.py"]
