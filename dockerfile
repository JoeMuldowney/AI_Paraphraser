FROM python:3.12-slim

WORKDIR Summarizer_App

COPY . .

# Install dependencies
RUN pip3 install torch torchvision torchaudio \
    && pip install --no-cache-dir -r app/requirements.txt


# Expose the port
EXPOSE 5000

# Run the app
CMD ["python", "app/app.py"]