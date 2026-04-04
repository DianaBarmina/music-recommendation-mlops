.PHONY: help install install-dev lint format test train push-data pull-data clean

help:
	@echo "Доступные команды:"
	@echo "  install      - установить зависимости"
	@echo "  install-dev  - установить dev зависимости"
	@echo "  lint         - проверить код"
	@echo "  format       - отформатировать код"
	@echo "  test         - запустить тесты"
	@echo "  train        - запустить DVC пайплайн"
	@echo "  push-data    - залить данные в DVC"
	@echo "  pull-data    - скачать данные из DVC"
	@echo "  clean        - очистить кэш"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

lint:
	flake8 src/ services/ tests/
	black src/ services/ tests/ --check
	isort src/ services/ tests/ --check-only

format:
	black src/ services/ tests/
	isort src/ services/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

train:
	dvc repro

push-data:
	dvc push

pull-data:
	dvc pull

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
