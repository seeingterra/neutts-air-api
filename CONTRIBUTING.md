# Contributing to NeuTTS Air API

Thank you for your interest in contributing to NeuTTS Air API! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR-USERNAME/neutts-air-api.git
cd neutts-air-api
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate.bat  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
pip install -r requirements-whisper.txt  # Optional
```

4. Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## Code Style

This project uses:
- **Black** for Python code formatting
- **Ruff** for linting
- **isort** for import sorting

These are enforced via pre-commit hooks. Run them manually with:
```bash
pre-commit run --all-files
```

## Testing

### Running the Test Suite

```bash
# Start the API server first
python api_server.py

# In another terminal, run tests
python test_api.py
```

### Manual Testing

1. Start the server:
```bash
./start_server.sh  # Linux/Mac
# or
start_server.bat   # Windows
```

2. Open the web GUI: http://127.0.0.1:8011/gui

3. Test the API endpoints:
```bash
# Health check
curl http://127.0.0.1:8011/health

# List voices
curl http://127.0.0.1:8011/voices

# Synthesize speech
curl -X POST http://127.0.0.1:8011/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice_id": "dave"}' \
  --output test.wav
```

## Making Changes

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following the code style guidelines

3. Test your changes thoroughly

4. Commit with descriptive messages:
```bash
git commit -m "Add feature: description of your feature"
```

5. Push to your fork:
```bash
git push origin feature/your-feature-name
```

6. Create a Pull Request

## Pull Request Guidelines

- **Descriptive title**: Use a clear, descriptive title
- **Description**: Explain what changes you made and why
- **Tests**: Include tests for new features
- **Documentation**: Update relevant documentation
- **Code style**: Ensure pre-commit hooks pass
- **Single focus**: Keep PRs focused on a single feature or fix

## Areas for Contribution

### High Priority
- Additional audio format support (MP3, OGG, etc.)
- Batch synthesis endpoints
- WebSocket streaming support
- Performance optimizations
- More comprehensive error handling
- Additional tests

### Nice to Have
- Docker support
- Authentication/authorization
- Rate limiting
- Metrics and monitoring
- Additional language support
- Mobile-optimized web GUI

### Documentation
- More usage examples
- API client libraries (Python, JavaScript, etc.)
- Deployment guides
- Troubleshooting guides
- Video tutorials

## Bug Reports

When reporting bugs, please include:
- **Description**: Clear description of the issue
- **Steps to reproduce**: Detailed steps to reproduce the bug
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, dependencies versions
- **Logs**: Relevant error messages or logs
- **Screenshots**: If applicable

## Feature Requests

When requesting features, please include:
- **Use case**: Why is this feature needed?
- **Description**: What should the feature do?
- **Examples**: Provide examples of how it would be used
- **Alternatives**: Have you considered alternatives?

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

## Questions?

- Open an issue for questions about contributing
- Check existing issues and PRs for similar discussions
- Read the documentation: README.md, API_README.md, WINDOWS_INSTALL.md

Thank you for contributing! 🎉
