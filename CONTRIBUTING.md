# Contributing to SmolaVision

Thank you for your interest in contributing to SmolaVision! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Poetry (recommended for dependency management)

### Setting Up the Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/smolavision.git
   cd smolavision
   ```

3. Install dependencies:
   ```bash
   # Using pip
   pip install -e ".[dev]"
   
   # Or using Poetry
   poetry install
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards

3. Write tests for your changes

4. Run the tests:
   ```bash
   pytest
   ```

5. Format your code:
   ```bash
   black .
   isort .
   ```

6. Commit your changes:
   ```bash
   git commit -m "Add your meaningful commit message here"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Create a pull request on GitHub

## Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all modules, classes, and functions
- Keep functions small and focused on a single responsibility
- Use meaningful variable and function names

## Project Structure

The project is organized into the following modules:

- `smolavision/video/`: Video processing functionality
- `smolavision/ocr/`: OCR processing functionality
- `smolavision/batch/`: Batch creation functionality
- `smolavision/models/`: AI model implementations
- `smolavision/tools/`: Tool implementations
- `smolavision/analysis/`: Analysis functionality
- `smolavision/pipeline/`: Pipeline implementations
- `smolavision/config/`: Configuration functionality
- `smolavision/cli/`: Command line interface
- `smolavision/output/`: Output generation functionality
- `tests/`: Test suite

## Testing

- Write unit tests for all new functionality
- Write integration tests for complex interactions
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage

## Documentation

- Update documentation for any changes to the API
- Add examples for new features
- Keep the README up to date
- Document any breaking changes

## Pull Request Process

1. Ensure your code follows the coding standards
2. Ensure all tests pass
3. Update documentation as needed
4. Fill out the pull request template
5. Request a review from a maintainer
6. Address any feedback from the review

## Release Process

1. Update the version number in `pyproject.toml`
2. Update the changelog
3. Create a new release on GitHub
4. Publish to PyPI

## Getting Help

If you need help with contributing, please:

- Open an issue on GitHub
- Reach out to the maintainers
- Check the documentation

Thank you for contributing to SmolaVision!
