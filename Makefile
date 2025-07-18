#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = bayesian_calibration
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	pip install -e .
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 bayesian_calibration
	isort --check --diff bayesian_calibration
	black --check bayesian_calibration

## Format source code with black
.PHONY: format
format:
	isort bayesian_calibration
	black bayesian_calibration



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	python3 -m venv .venv
	@echo ">>> New virtualenv created. Activate with: source .venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
