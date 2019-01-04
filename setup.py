import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="KTBoost",
    version="0.0.6",
    author="Fabio Sigrist",
    author_email="fabiosigrist@gmail.com",
    description="Implements several boosting algorithms in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fabsig/KTBoost",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)