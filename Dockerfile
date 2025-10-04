# 1. Base Image: Use a stable, slim version of Python for smaller image size
FROM python:3.9-slim-buster

# 2. Set Working Directory: All subsequent commands run inside this directory
WORKDIR /app

# 3. Copy Requirements: Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# 4. Install Dependencies: Use system dependencies first, then pip install
#    gunicorn is installed here to serve the Flask app
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Remaining Code: Copy the entire project code into the container
#    This includes src/ components, app.py, etc.
COPY . .

# 6. Expose Port: The port Gunicorn/Flask will listen on
EXPOSE 5000

# 7. Entrypoint/Command: Define how the application starts
#    Use Gunicorn for production serving (4 workers = good starting point)
#    'app:app' refers to the Flask instance named 'app' inside the 'app.py' file.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]