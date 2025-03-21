run:
	python src/app.py

run-test2:
	python src/tests/spec_2_linear_speed.py

install:
	pip install -r requirements.txt

save-deps:
	pip freeze > requirements.txt