# Use the official Python image as a base
#FROM python:3.10
FROM python:3.10-slim
#FROM python:3.10-slim


# Set the working directory
WORKDIR /app	

# Concat to reduce # of layers

RUN apt-get update && \
    apt-get install -y python3-venv && \
    pip install -U pip && \
    pip install --upgrade pip && \
    pip install -U setuptools wheel && \
    pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install autogluon && \
    pip install --no-cache-dir -r requirements.txt && \
    python3 -m venv soilmoisture_env
	
#upgrade pip
#RUN pip install --upgrade pip && \


# Install virtualenv
#RUN pip install --upgrade pip && \
#    pip install --no-cache-dir -r requirements.txt


# Create and activate virtual environment
#RUN python3 -m venv soilmoisture_env


# Activate virtual environment
ENV PATH="./app/soilmoisture_env/bin:$PATH"



# Copy the requirements file into the container at /app
#COPY requirements.txt /app/


# Install any needed packages specified in requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Specify the command to run on container start
CMD ["python", "app.py"]