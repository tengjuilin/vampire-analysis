import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vampire-analysis",
    version="0.1.0.dev10",
    author="Teng-Jui Lin",
    author_email="lintengjui@outlook.com",
    description="VAMPIRE (Visually Aided Morpho-Phenotyping Image Recognition) analysis quantifies and visualizes heterogeneity of cell and nucleus morphology.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tengjuilin/vampire-analysis",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'opencv-python',
        'scikit-image',
        'scikit-learn',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
