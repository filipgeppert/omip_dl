FROM python:3.6

ENV PYTHONUNBUFFERED 1
ARG dirname=src
# create root directory for our project in the container
RUN mkdir /$dirname

# Set the working directory to /dirname
WORKDIR /$dirname

# Copy the current directory contents into the container at /dirname
COPY . /$dirname/

# Install dependencies
RUN pip install pipenv
COPY Pipfile Pipfile.lock /$dirname/
RUN pipenv install --system