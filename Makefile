

setup-venv:
	@echo "Setting up virtual environment..." &&
	@python3 -m venv .venv &&
	@source ./.venv/bin/activate &&
	@pip install --upgrade pip &&
	@pip install -r requirements.txt
	@pip install -r test_requirements.txt
	@pip install "cvxpy[ecos]"

unit-tests:
	@echo "Running unit tests..." && \
	source ./.venv/bin/activate && \
	pytest -v