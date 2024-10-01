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

### Docstrings Style
In order to fit the Sphinx's automatic docstring generation, we recommend to document the functions in the default [Sphinx style](https://www.sphinx-doc.org/en/master/tutorial/describing-code.html#documenting-python-objects), for example:
``` python
def symmetric_tree(h : int, degree : int, new_node : TreeNode = TreeNode, fake_root : TreeNode = None) -> HypTree:
    """ Generate tree of given height and degree

    A tree of height zero contains just the root;
    a tree of height one contains the root and one level of leaves below it,
    and so forth.

    :param h: The height of the tree
    :param degree: The degree of each node in the tree
    :param new_node: The node used to construct the tree, defaults to TreeNode
    :param fake_root: The fake root node, defaults to None
    :raises ValueError: If height is negative
    :return: The constructed tree
    """
    if h < 0:
        raise ValueError(f'Height shall be nonnegative integer, received {h=}.')

    def _builder(h: int, degree: int, parent):
        node = new_node(); node.parent = parent; node.children = ChildList()
        if h > 1:
            node.children = ChildList([_builder(h - 1, degree, node) for _ in range(degree)])
        return node

    if fake_root is None:
        return HypTree(root=_builder(h + 1, degree, None))
    else:
        fake_root.children = ChildList([_builder(h + 1, degree, fake_root)])
        return HypTree(root=fake_root)
```
If you are using VScode, it is easy to generate such template automatically using the [autoDocstring extension](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring). After installation, open the extension settings and choose `sphinx-notypes` under `Docstring Format` tab.

### Testing

- Write tests for new features and bug fixes.
- Make sure all existing tests pass before submitting changes.
- Run the test suite locally before submitting your PR: `make test`.

### Documentation

- Document all public classes, methods, and functions using clear and concise comments and docstrings.
- Update the README.md file with any relevant changes to the project, including installation and usage instructions.

### Build Documentation Locally
The documentation compiling replies on _sphinx_ package, to install it, use _pip_:
``` bash
$ pip install sphinx
```
After install _sphinx_, check if there are three major files under `./docs` directory: `make.bat`, `Makefile` and `source/config.py`. If not, run
``` bash
$ sphinx-quickstart
```
to automatically generate these files, and add manually
``` python
import os
import sys
sys.path.insert(0, os.path.abspath('../../hyperiax/'))
``` 
into `source/config.py` in the beginning to configure the directory for _sphinx_ to find the source code.

You may also need to install additional packages
* _sphinx-rtd-theme_: to use the readthedoc theme) 
* _sphinx-autodoc-annotation_: for automatic type annotation). 
* _nbsphinx_: for compiling jupyter notebooks
* _pandoc_: addons for _nbsphinx_
To install them except _pandoc_, use pip:
``` bash
$ pip install sphinx_rtd_theme
$ pip install sphinx-autodoc-annotation
$ pip install nbsphinx
```
To install _pandoc_, in MacOS, using _brew_:
``` bash
$ brew install pandoc
```

After everything is installed, you can compile the documentation locally:
``` bash
$ cd ./docs
$ make clean
$ make html
```
This should generate compiled html files under `./docs/build/` directory.