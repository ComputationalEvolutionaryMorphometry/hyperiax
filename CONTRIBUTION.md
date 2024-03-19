# Contributing Guidelines

Thanks for taking time to contribute!👍🎉

When contributing to the repository, we suggest a general guideline for easily organizing and integrating cool ideas from everyone, so please follow it in your interactions with the project.


## Setting up the project locally

1. [Fork the repository on GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo), and clone your forked repository to your local machine.
```
# Clone the forked repository
https://github.com/<your username>/hyperiax

# Navigate to the directory
cd hyperiax
```
2. [Create the environment and install required packages](./README.md#installation)
3. Make sure your `main` branch points at the original repository's `main`, so that you can easily get updates from the original by just running `git pull`: 
```
# Add the original repository as a remote to your own repository
git remote add upstream https://github.com/ComputationalEvolutionaryMorphometry/hyperiax

# Fetch the upstream and connect your local main branch to the original main
git fetch upstream
git branch --set-upstream-to=upstream/main main

# Pull the latest updates from the original repository
git pull
```

4. Create a new branch for your new development and testing (Recommended)
```
# Create a new local branch called 'my-dev'
git checkout -b my-dev

# ... (Your contributions)

# Test your changes
python -m pytest

# Commit your changes
git commit -am "Add my features"
# Push them to your remote repository
git push origin my-dev
```

5. If everything is ready, consider [submitting a pull request (PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request?tool=webui) against the `main` branch of the original repository.

## Pull Request Guidelines

### PR Philosophy
- Ensure your PR is focused on a single change and has a clear title and description.
- Check the issues and include relevant issue numbers in the PR description if applicable.
- Make sure all tests pass before submitting the PR.
- Keep your PR up to date with the latest changes in the `main` branch. Do this by pulling the upstream branch on your fork.

### Code Style
- Use clear and descriptive variable and function names.
- Write docstrings for all public classes, methods, and functions.
- Do not use any stateful randomness. ie. a PRNGKey should only be passed to a constructor if it is immediately disposed.

### Testing

- Write tests for new features and bug fixes.
- Make sure all existing tests pass before submitting changes.
- Run the test suite locally before submitting your PR: `make test`.

### Documentation

- Document all public classes, methods, and functions using clear and concise comments and docstrings.
- Update the README.md file with any relevant changes to the project, including installation and usage instructions.
