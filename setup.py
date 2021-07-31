import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vampire",
    version="0.0.dev1",
    author="Teng-Jui Lin",
    author_email="lintengjui@outlook.com",
    description="Visually Aided Morpho-Phenotyping Image Recognition, a robust method to quantify cell morphological heterogeneity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tengjuilin/vampire",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
