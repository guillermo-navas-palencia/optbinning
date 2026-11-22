# Contributing to optbinning

Thank you for considering a contribution. This guide explains how to propose
changes so they can be reviewed quickly and merged.

## Discuss before you code

- Open a GitHub issue describing the bug or feature before starting work on a
  pull request (PR). Wait for maintainer feedback on the approach.
- PRs without a prior discussion will be closed automatically.
- optbinning is a generic library. Additions that are specific to a single
  dataset, company, or workflow, or that fall outside the library's scope, will
  be discarded. When in doubt, propose the idea in an issue first.

## Branches and pull requests

- `develop` is the main development branch. **All PRs must target `develop`.**
- Create a branch from an up-to-date `develop`, using a descriptive name such
  as `feat/<short-description>` or `fix/<short-description>`.
- Split large changes into small PRs, each focused on one topic (for example, a
  refactor, then the feature, then its docs). Each PR should be easy to review
  on its own and should pass CI.
- In the PR description, explain what changes and why, link the related issue,
  and describe how you tested the change.
- Write clear commit messages in the imperative mood, for example
  `Add sample_weight support to continuous binning`.

## Code style

- Target Python 3.11 and newer.
- Follow flake8 with a maximum line length of 79 characters for new code. Run
  `flake8` on the files you changed. Do not reformat unrelated existing code.
- Type hints are optional. If you add them, match the style of the surrounding
  module.
- Write docstrings in NumPy style (`Parameters` and `Returns` sections), as used
  in `optbinning/binning/binning.py`. Use an imperative summary line, such as
  `Return the bin edges.` rather than `Returns the bin edges.`.
- Follow the import style of the code base. Group imports in this order, with a
  blank line between groups: standard library, third-party packages, and
  relative imports within `optbinning`. Use relative imports inside the package
  and prefer one imported name per `from` line, as in
  `optbinning/binning/binning.py`.
- Reuse existing helpers and follow the design patterns already in the code
  base. Do not add new abstractions, public APIs, or runtime dependencies
  without discussing them in an issue first.
- Keep code and comments concise. Avoid verbose code, dead code, redundant
  wrappers, and boilerplate.
- Comment only code that needs clarification.
- Do not reference GitHub issue numbers in code comments. Put them in the
  commit message or PR description instead.

## Use of AI tools

AI tools are welcome as development aids. They do not change who is responsible
for a contribution: the contributor who opens a PR is accountable for every line
in it.

- Understand every line you submit. You should be able to explain and defend
  each change during review. Do not submit code you have not read, run, and
  understood.
- Keep PRs small and focused (see [Branches and pull requests](#branches-and-pull-requests)).
  Large PRs that touch many files or fix several problems at once are hard to
  review and maintain.
- Check readiness before requesting review: run the test suite and flake8, test
  edge cases, and confirm the behavior manually.
- Make sure the design matches existing patterns in the code base. Reject
  generated abstractions that duplicate existing functionality. Remove verbose
  code, unnecessary comments, unused imports, and APIs that do not exist in the
  library or its dependencies.
- Review AI-generated text in issues and PR descriptions before posting it.

### Mathematical formulations and core algorithms

The mathematical formulations behind the optimization models (for example, the
MIP, CP, and LS formulations in `optbinning/binning/`) and the statistical
methods are owned and maintained by the project maintainer. Do not submit
AI-generated changes to these areas without a thorough understanding of the
existing implementation. Propose changes in an issue with the technical
justification first.

### Priorities and roadmap

- The maintainer defines the project's direction, priorities, and release scope.
  Planned work is tracked in the project's GitHub Projects boards. Wait for
  confirmation of scope and target release before opening a PR.
- Bug reports from users have the highest priority. Improvements with limited
  impact on users may be deferred, even when they are technically valid.
- Speculative improvements, refactors without a clear need, and bulk submissions
  of unrelated changes may be declined, even when they are correct.
- The maintainer may decline a contribution that does not fit the project's
  priorities or roadmap, regardless of its quality.

## Unit tests

- Tests live in `tests/` and use pytest. Name files `test_*.py` and add tests to
  the existing file for the module you changed where possible.
- Add unit tests for every new behavior and every bug fix. A bug-fix test should
  fail without the fix.
- Cover edge cases with focused tests. Do not write tests that only restate the
  implementation or check constants and defaults.
- Set up the environment as in CI:

  ```bash
  pip install -r test_requirements.txt
  pip install -e .[distributed,test,ecos]
  ```

- Run the tests with `pytest` for the full suite, or `pytest tests/test_binning.py`
  for a single file.

## Examples and documentation

- New functionality needs a notebook example in `doc/source/tutorials/`.
  - For a small extension, add a section to the most relevant existing tutorial.
  - For a large feature or example, add a new notebook, for example
    `tutorial_<topic>.ipynb`.
- Update the docstrings and any directly related documentation in the same PR.

## Review process

- Maintainers may ask you to split a PR, narrow its scope, or change its
  approach. They may close PRs that are out of scope.
- Respond to review comments promptly and keep discussions respectful and
  focused on the code.

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0, the license of this project.
