FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training notebook (reference only)
COPY training/tranformation.ipynb  ./training/Adaptation.ipynb

# Explicitly state that this container is for reference
CMD ["echo", "Training notebook  tranformation and Adaptation included for reference. the file is not at all automated because i had to traine the model locally and then log the metrics and its parameteres on ML Flow Please run locally to train models."]
