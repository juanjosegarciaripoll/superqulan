init:
	pip install -r requirements.txt

install:
	pip install .

check:
	cd tests && python -m unittest tests -v
