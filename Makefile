run:
	python src/app.py

install:
	pip install -r requirements.txt

save-deps:
	pip freeze > requirements.txt