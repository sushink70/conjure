# Contributing to Conjure: Python Code Visualizer

Thank you for considering contributing to **Conjure**! We're excited to have you on board. Whether it's a bug fix, new DSA hint, feature addition, or documentation tweak, your input helps make this tool more powerful for Python learners and DSA enthusiasts everywhere. 🌟

This guide outlines how to get started, our expectations for contributions, and best practices to keep things smooth.

## Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold it. If you encounter issues, reach out to the maintainers via [GitHub Issues](https://github.com/sushink70/conjure/issues).

## Getting Started

### Prerequisites
- Python 3.8+ (tested on 3.12)
- Git
- [Rich library](https://rich.readthedocs.io/) (`pip install rich`)

### Fork & Clone
1. Fork the repository on GitHub: [sushink70/conjure](https://github.com/sushink70/conjure).
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/conjure.git
   cd conjure
   ```
3. Add the upstream remote (for syncing changes):
   ```bash
   git remote add upstream https://github.com/sushink70/conjure.git
   ```

### Development Setup
1. Install dependencies (just Rich for now):
   ```bash
   pip install -r requirements.txt  # Create this if needed, or pip install rich
   ```
2. Make the script executable (optional, for local runs):
   ```bash
   chmod +x conjure.py
   ```
3. Test the setup with the built-in example:
   ```bash
   python conjure.py --summary-only
   ```
   - You should see the Bubble Sort summary without errors.

### Branching Strategy
- Create a feature branch from `main`:
  ```bash
  git checkout -b feat/your-feature-name  # e.g., feat/graph-bfs-hint
  ```
- Keep branches small and focused (one change per PR).

## Making Changes

### Code Style
- Follow [PEP 8](https://peps.python.org/pep-0008/) for formatting.
- Use [Black](https://black.readthedocs.io/) for auto-formatting:
  ```bash
  pip install black
  black conjure.py
  ```
- Docstrings: Use Google or NumPy style for new functions/classes.
- Type hints: Add them where possible (e.g., `def func(x: int) -> str:`).
- Comments: Explain *why*, not *what*—assume readers know Python basics.

### Key Areas for Contributions
- **New Hints**: Extend `_generate_hint()` for patterns like sliding windows, union-find, or tree traversals.
- **Complexity Detection**: Improve `detect_complexity()` with more AST walkers.
- **Visuals**: Add new renderings in `format_value()` (e.g., tree diagrams for dicts).
- **Error Handling**: Enhance suggestions in `execute()`.
- **Examples**: Add DSA scripts to `/examples/` (e.g., quicksort, Dijkstra).
- **Docs**: Update README or add user guides.

### Testing Your Changes
- **Manual Testing**: Run with various flags:
  ```bash
  python conjure.py examples/bubble_sort.py --auto --delay 0.1
  python conjure.py your_new_example.py --summary-only
  ```
- **Edge Cases**: Test recursion (e.g., deep fib), loops, errors, and large inputs (up to max-steps).
- **No Formal Tests Yet**: We use manual verification. If you add features, include a simple repro script in `/examples/`.
- **Version Bumps**: Update the version in the docstring (e.g., "v2.2") and changelog in README.

### Commit Messages
Use conventional commits for clarity:
- `fix: resolve infinite loop in trace_calls`
- `feat: add union-find hint detection`
- `docs: update installation steps`
- Keep messages <72 chars; reference issues with "Fixes #123".

## Submitting Issues
- Use the [issue template](https://github.com/sushink70/conjure/issues/new) for bugs/features.
- Include:
  - Steps to reproduce (code snippet + command).
  - Expected vs. actual behavior.
  - Environment (Python version, OS).
- For features: Describe the problem it solves and a rough implementation idea.

## Submitting Pull Requests (PRs)
1. Push your branch:
   ```bash
   git push origin feat/your-feature-name
   ```
2. Open a PR against `main`:
   - Use a clear title (e.g., "feat: Detect sliding window patterns").
   - Describe changes, motivation, and testing done.
   - Link to related issues.
3. We'll review promptly—expect feedback within a few days.

### PR Guidelines
- **Small PRs**: Aim for <300 lines changed.
- **Squash Commits**: Use `git rebase -i` for clean history.
- **CI/CD**: No automated CI yet—manual review covers it.
- **Releases**: Merges to `main` trigger informal releases (bump version + tag).

## Roadmap & Ideas
From the README—feel free to tackle these:
- Web/HTML export for traces.
- Jupyter integration.
- More DSA detectors (e.g., segment trees, A* search).
- Benchmarking for performance hints.
- Multi-language support (start with JS?).

Got an idea not listed? Open an issue first!

## Questions?
- Ping us in [Issues](https://github.com/sushink70/conjure/issues) or Discussions.
- Join the conversation on X (formerly Twitter) via @sushink70 or #ConjureViz.

Happy contributing—let's conjure some magic! 🐍✨

---

*Last updated: December 20, 2025*