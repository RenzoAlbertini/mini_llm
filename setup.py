from setuptools import find_packages, setup


setup(
    name="mini_llm",
    version="1.0.0",
    description="MiniLLM - Lightweight Local Language Model",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    py_modules=["mini_llm"],
    packages=find_packages(
        include=[
            "agents",
            "agents.*",
            "benchmark",
            "benchmark.*",
            "chat",
            "chat.*",
            "inference",
            "inference.*",
            "model",
            "model.*",
            "plugins",
            "plugins.*",
            "tests",
            "tests.*",
            "tokenizer",
            "tokenizer.*",
            "training",
            "training.*",
            "utils",
            "utils.*",
        ]
    ),
    install_requires=["torch", "fastapi", "uvicorn", "requests"],
    python_requires=">=3.10",
    license="MIT",
    entry_points={"console_scripts": ["mini-llm=cli:main"]},
)
