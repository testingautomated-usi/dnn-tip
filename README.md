# Python Package Starter
![test](https://github.com/vikpe/python-package-starter/workflows/test/badge.svg?branch=master) [![codecov](https://codecov.io/gh/vikpe/python-package-starter/branch/master/graph/badge.svg)](https://codecov.io/gh/vikpe/python-package-starter) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features
* Multi python version support ([pyenv](https://github.com/pyenv/pyenv), [tox](https://github.com/tox-dev/tox/))
* Single config using the new standardized `pyproject.toml` ([PEP518](https://www.python.org/dev/peps/pep-0518/))
* Simple build/publish/dependency management using [poetry](https://github.com/sdispater/poetry)
* Continous integration ([GitHub Actions](https://help.github.com/en/actions))
* Code coverage reporting ([Codecov](https://codecov.io/))
* Code formatting ([black](https://github.com/psf/black))

## Prerequisites
* [pyenv](https://github.com/pyenv/pyenv) (for automatic installation use [pyenv-installer](https://github.com/pyenv/pyenv-installer))
* [poetry](https://github.com/sdispater/poetry)

## Installation
1. Install [pyenv](https://github.com/pyenv/pyenv).
2. Install the Python versions you want to support using `pyenv`.
  ```sh
  pyenv install 3.6.9
  pyenv install 3.8.0
  ```
3. Clone repo: `git clone git@github.com:vikpe/python-package-starter.git [PACKAGE_NAME]` 
4. `cd [PACKAGE NAME]`
5. Create a virtual env: `pyenv virtualenv 3.8.0 foo38`
6. Activate virtual env: `pyenv activate foo38`
7. Install poetry: `pip install poetry`
8. Install dependencies: `poetry install`
9. Edit `pyproject.toml`, update project name, description and author and any other settings you like.

## Usage

Command | Description
--- | ---
`poetry add [package]` | Add package to dependencies.
`poetry add -D [package]` | Add package to dev dependencies.
`poetry run pytest` | Run tests in local Python version.
`poetry run ptw tests foo --clear` | Watch for file changes and run tests in local Python version.
`poetry run tox` | Run tests in all Python versions defined in `tox.ini`.
`poetry run black .` | Run black code formatter.
`poetry build` | Build sdist and wheel to `/dist`.
`poetry publish` | Publish package to PyPi.

## Continous integration

### GitHub Actions
Tests are run whenever there is a commit, see `.github/workflows/test.yml` for details.

### Code coverage
Enable code coverage reporting to [Codecov](https://codecov.io/) by creating a secret with name `CODECOV_TOKEN` in your repository settings (Settings -> Secrets -> New sectret) and value set to the token created by Codecov.
