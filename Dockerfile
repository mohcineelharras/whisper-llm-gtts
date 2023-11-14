# Use an official Python runtime as a parent image
FROM python:3.11

# Install system packages
RUN apt-get update && apt-get install -y \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
