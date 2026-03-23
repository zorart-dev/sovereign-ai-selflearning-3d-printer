# Contributing to Sovereign-v5.0

Thank you for your interest in contributing to Sovereign! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help each other learn and improve
- Report issues professionally

## How to Contribute

### 1. Report Bugs
- **Check existing issues** before reporting
- **Provide details**: Python version, PyTorch version, error message
- **Include logs**: Attach sovereign.log or training_log.json
- **Reproducible steps**: Clear instructions to reproduce the issue

### 2. Suggest Features
- **Search issues** to avoid duplicates
- **Describe use case**: Why do you need this feature?
- **Provide examples**: Show how it would be used
- **Consider scope**: Does it fit the project goals?

### 3. Submit Code

#### Setup Development Environment
```bash
# Clone repository
git clone https://github.com/yourusername/sovereign-v5.0.git
cd sovereign-v5.0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8 mypy sphinx
```

#### Code Style Guidelines

Follow **PEP 8**:
```bash
# Format code with Black
black sovereign_v5_final/

# Check style with Flake8
flake8 sovereign_v5_final/

# Type checking
mypy sovereign_v5_final/
```

**Code Standards**:
- 4 spaces indentation
- Max 80 characters per line
- Descriptive variable names
- Docstrings for all public functions
- Type hints throughout
- Comments for complex logic

#### Testing

Write tests for new features:

```python
# Example: test_my_feature.py
import pytest
from my_module import MyClass

def test_my_feature():
    """Test description"""
    # Arrange
    obj = MyClass()
    
    # Act
    result = obj.do_something()
    
    # Assert
    assert result == expected_value
```

Run tests:
```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_sovereign.py::TestMyFeature -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

#### Commit Messages

Follow conventional commits:
```
type(scope): subject

body

footer
```

Examples:
- `feat(agent): Add PPO epsilon decay`
- `fix(hardware): Fix OctoPrint connection timeout`
- `docs(config): Update parameter reference`
- `test(reward): Add vision reward tests`

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

#### Pull Request Process

1. **Create a branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes** and commit
   ```bash
   git commit -m "feat(module): Add new feature"
   ```

3. **Write/update tests**
   ```bash
   pytest tests/ -v
   ```

4. **Format and lint**
   ```bash
   black .
   flake8 .
   ```

5. **Update documentation**
   - README.md if needed
   - Docstrings in code
   - CHANGELOG if applicable

6. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   ```

7. **PR Description**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Related Issues
   Fixes #123
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   
   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing done
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Comments added for clarity
   - [ ] Documentation updated
   - [ ] No new warnings
   ```

## Contribution Areas

### High Priority
- Bug fixes and stability improvements
- Performance optimizations
- Documentation improvements
- Test coverage expansion

### Welcome Additions
- New printer hardware support
- Custom reward functions
- Safety constraint implementations
- Visualization tools
- Data collection utilities

### Future Features
- Real federated learning with secure aggregation
- Web dashboard for monitoring
- Model compression for edge deployment
- Sim-to-real transfer learning
- Advanced curriculum learning

## Documentation

### Writing Documentation

Use consistent style:
```markdown
# Heading 1

## Heading 2

**Bold** for emphasis

`code` for inline code

\`\`\`python
# Code blocks with language
def example():
    return "code"
\`\`\`

- Bullet points
- For lists

1. Numbered
2. Lists
```

### Documentation Files

- `README.md`: Overview and quick start
- `DEVELOPER_GUIDE.md`: Architecture and design
- `CONFIG_GUIDE.md`: Configuration reference
- `QUICK_REFERENCE.md`: Commands and snippets
- In-code docstrings: Implementation details

## Review Process

### For Contributors
- Address all feedback
- Keep discussions professional
- Ask questions if unclear
- React positively to suggestions

### For Reviewers
- Be constructive and kind
- Ask clarifying questions
- Suggest improvements
- Acknowledge good work

## Development Workflow

```
Feature Request/Issue
    ↓
Create Branch
    ↓
Implement & Test
    ↓
Submit PR
    ↓
Code Review
    ↓
Merge to Main
    ↓
Release
```

## Reporting Security Issues

⚠️ **Do not** open public issues for security vulnerabilities.

Email security details to the maintainers with:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Performance Considerations

When contributing:
- Profile code for bottlenecks
- Optimize critical paths
- Consider memory usage
- Test on different hardware

## Questions?

- **Usage**: Check README.md and examples.py
- **Architecture**: See DEVELOPER_GUIDE.md
- **Configuration**: Review CONFIG_GUIDE.md
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md
- Release notes
- GitHub contributors page

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Sovereign! Together, we're building better autonomous systems. 🚀
