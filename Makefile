run:
	python src/app.py

run-test2:
	python src/tests/spec_2_linear_speed.py

run-test_vision:
	python -m src.tests.test_vision
	
install:
	pip install -r requirements.txt

save-deps:
	pip freeze > requirements.txt
