import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sevir", 
    version="0.1",
    author="Mark Veillette",
    author_email="mark.veillette@mit.edu",
    description="Tools for working with the SEVIR dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MIT-AI-Accelerator/eie-sevir",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
