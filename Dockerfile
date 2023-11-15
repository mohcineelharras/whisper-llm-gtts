# Dockerfile for Streamlit app
FROM python:3.11

# Update the package index and install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0

# Create a directory for your application inside the container
WORKDIR /app

# Copy the requirements file from the parent directory into the container
COPY requirements.txt /app/

# Copy the rest of the project files and directories from the parent directory into /app/ in the container
COPY . /app/

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port that Streamlit will use (8501 by default)
EXPOSE 8501  

# Define the command to run when the container starts
CMD ["streamlit", "run", "app.py"]
