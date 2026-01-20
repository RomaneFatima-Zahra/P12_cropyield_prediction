# Base image avec Python
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer Poetry
RUN pip install --no-cache-dir poetry

# Copier les fichiers de dépendances
COPY pyproject.toml poetry.lock ./

# Installer les dépendances (sans créer de virtualenv)
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main


# Copier tout le code
COPY . .

# Exposer le port FastAPI
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

