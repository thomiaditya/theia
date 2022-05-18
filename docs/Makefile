.DEFAULT_GOAL := default
app_root = backend/app
pkg_src = $(app_root)/app
tests_src = $(app_root)/tests
local_tests_src = $(app_root)/tests/local

isort = isort -rc $(pkg_src) $(tests_src)
black = black $(pkg_src) $(tests_src)
flake8 = flake8 $(pkg_src) $(tests_src)
mypy = mypy $(pkg_src)
mypy_tests = mypy $(pkg_src) $(tests_src)


.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: check-format
check-format:
	$(isort) --check-only
	$(black) --check

.PHONY: lint
lint:
	$(flake8)

.PHONY: mypy
mypy:
	$(mypy)

.PHONY: mypy-tests
mypy-tests:
	$(mypy_tests)

.PHONY: test-local
test-local:
	pytest $(local_tests_src) --cov=$(pkg_src)

.PHONY: test-dev
test-dev:
	./scripts/test-dev.sh

.PHONY: test-deploy
test-deploy:
	./scripts/test-deploy.sh

# Consider using test-dev or test-deploy instead
.PHONY: testcov
testcov:
	pytest $(local_tests_src) --cov=$(pkg_src)
	@echo "building coverage html"
	@coverage html
	@echo "opening coverage html in browser"
	@open htmlcov/index.html

.PHONY: static
static: format lint mypy

.PHONY: default
default: static test-local

# Consider test-dev or test-deploy instead
.PHONY: verify
verify: check-format lint mypy test-local

.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -rf `find . -type d -name '*.egg-info' `
	rm -rf `find . -type d -name 'pip-wheel-metadata' `
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build

.PHONY: lock
lock:
	./scripts/setup-lock.sh

.PHONY: develop
develop:
	./scripts/setup-develop.sh

.PHONY: poetryversion
poetryversion:
	cd $(app_root); poetry version $(version)

.PHONY: version
version: poetryversion
	$(eval NEW_VERS := $(shell cat $(app_root)/pyproject.toml | grep "^version = \"*\"" | cut -d'"' -f2))
	@sed -i "" "s/__version__ = .*/__version__ = \"$(NEW_VERS)\"/g" $(pkg_src)/__init__.py