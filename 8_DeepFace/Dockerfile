# Użyj oficjalnego obrazu Python 3.8
FROM python:3.10

# Ustaw katalog roboczy w kontenerze
WORKDIR /app

# Skopiuj pliki projektu do katalogu /app w kontenerze
COPY . /app

# Zainstaluj zależności
RUN pip install PySide6 deepface

# Polecenie do uruchomienia aplikacji
CMD ["python", "DeepFace.py"]