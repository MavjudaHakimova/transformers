# Transformer Language Classifier

Этот проект реализует классификатор языков на основе модели трансформера.

## Требования

- Docker

## Установка и запуск

1. Склонируйте репозиторий:

   ```bash
   git clone <URL_вашего_репозитория>
   cd transformer_language_classifier

2. Постройте Docker-образ:
```
docker build -t transformer_language_classifier .
```
Запустите контейнер:
```
docker run --rm transformer_language_classifier
```