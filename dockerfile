# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app.py /app
COPY requirements.txt /app
COPY uploads /app
COPY filenames_products.pkl /app
COPY features_list_for_prods.pkl /app
COPY data /app/data

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8509

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["streamlit", "run", "--server.port", "8509", "app.py"]
