from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="risk-neutral-density",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Comprehensive risk-neutral density extraction and analysis from options data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/risk-neutral-density",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "sphinx>=4.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
