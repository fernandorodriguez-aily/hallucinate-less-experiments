> **IMPORTANT** This project relies on [Aily-Labs](https://github.com/Aily-Labs) code

This repository contains code for executing experiments to evaluate a model's ability to discern if a given context is sufficient to answer a specific question. The project uses closed-source LLMs to classify the relevance of context passages in relation to given questions.

## Project Overview

The main purpose of this project is to:

1. Analyze the relevance of context passages to specific questions.
2. Classify contexts into three categories: Highly Relevant, Relevant, or Irrelevant.
3. Evaluate the model's performance in discerning context relevance.

## Setup

This project uses UV as the package manager. To set up the project:

1. Add Cloudsmith URL from `.env` into TOML (to avoid publicly exposing it)

For Linux:
```shell
sed -i '/\[tool\.uv\]/,/^$/c\[tool.uv]\nindex-url = "'$(grep CLOUDSMITH_URL .env | cut -d '=' -f2)'"' pyproject.toml
```

For macOS:
```shell
sed -i '' '/\[tool\.uv\]/,/^$/c\
[tool.uv]\
index-url = "'$(grep CLOUDSMITH_URL .env | cut -d '=' -f2)'"
' pyproject.toml
```

> Since this adds the Cloudsmith URL to the TOML, we have created an extra pre-commit step to remove it to avoid forgetting and publicly sharing that information.

2. Install UV if you haven't already:

```shell
curl -sSf https://astral.sh/uv/install.sh | sh
```

3. Create a virtual environment and install dependencies

```shell
uv sync
```

## Usage

To run the main script:

```shell
python src/run.py
```

This will execute the context classification process using the specified model and dataset in the `config.yaml` file.

This will execute the context classification process using the specified model and dataset in the config.yaml file. As a result, a SQLite database will be created with the LLM's outputs. The database structure is as follows:

```sql
CREATE TABLE IF NOT EXISTS results (
    id INTEGER,
    file TEXT,
    model TEXT,
    title TEXT,
    reasoning TEXT,
    structured_reasoning TEXT,
    relevance_label TEXT,
    relevance_label_gt TEXT,
    PRIMARY KEY (id, file, model)
)
```

This database stores the classification results, including the model's reasoning, structured output, and relevance labels for each processed item. The combination of `id`, `file`, and `model` serves as the primary key to uniquely identify each entry.

## Configuration

- `config.yaml`: Contains configuration settings for the project.
- `prompts.yaml`: Stores the prompts used for context classification.

## Key Components

- `src/run.py`: Main script for running the classification process.
- `src/database.py`: Handles database operations.
- `src/schema.py`: Defines the Pydantic schema for classification output.
- `src/utils.py`: Contains utility functions for setup and data handling.

## Contributing

In order to contribute you need to install `pre-commit`:

```shell
pre-commit install
```
