from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from aily_ai_brain.common.enums import BedrockModelID, OpenAIModelID, OpenRouterModelID
from aily_ai_brain.langfuse.langfuse_handler import get_langfuse_handler
from aily_py_commons.aily_settings import AilySettings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from tqdm import tqdm

from src.database import get_id_file_model_combinations, initialize_db, insert_row
from src.schema import ClassificationOutput
from src.utils import load_yaml, setup_basic_chain


def process(
    analyze_chain: Runnable,
    format_chain: Runnable,
    context: str,
    question: str,
    so_format_instructions: str,
    langfuse_handler: BaseCallbackHandler,
) -> Union[Any, Any]:
    """
    Process the input using analyze and format chains.
    Args:
        analyze_chain: Chain for analysis.
        format_chain: Chain for formatting.
        context: Input context.
        question: Input question.
        so_format_instructions: Formatting instructions.
        langfuse_handler: Callback handler for logging.
    Returns:
        Tuple of raw output and formatted output.
    """
    # Invoke the analyze chain
    output = analyze_chain.invoke(
        input={"context": context, "question": question},
        config={"callbacks": [langfuse_handler]},
    )

    # Invoke the format chain
    formatted_output = format_chain.invoke(
        input={
            "input": output,
            "pydantic_format_instructions": so_format_instructions,
        },
        config={"callbacks": [langfuse_handler]},
    )

    return output, formatted_output


def run(
    data_path: Union[str, Path],
    db_path: Union[str, Path],
    model_id: Union[OpenAIModelID, BedrockModelID, OpenRouterModelID],
    max_rows: Optional[int] = None,
    max_workers: Optional[int] = 4,
):
    """
    Run the main processing pipeline.
    Args:
        data_path: Path to the input data file.
        db_path: Path to the database.
        model_id: ID of the model to use.
        max_rows: Maximum number of rows to process (optional).
        max_workers: Maximum number of worker threads (optional).
    """
    # Convert data_path and db_path to Path objects
    data_path = Path(data_path)
    db_path = Path(db_path)

    # Set up Langfuse handler
    langfuse_handler = get_langfuse_handler(
        langfuse_tags=["hallucinate_less", "research", "mila", "genai"],
    )

    # Set up analyze chain
    analyze_chain = setup_basic_chain(
        prompt=prompts["classify"],
        model_id=model_id,
        langfuse_handler=langfuse_handler,
        max_tokens=config["max_tokens"],
        sensitive_data=False,
    )

    # Set up structured output parser and chain
    so_parser = PydanticOutputParser(pydantic_object=ClassificationOutput)
    so_chain = setup_basic_chain(
        prompt=prompts["structured_output"],
        model_id=model_id,
        langfuse_handler=langfuse_handler,
        max_tokens=config["max_tokens"],
        output_parser=so_parser,
        sensitive_data=False,
    )

    # Initialize database
    initialize_db(db_path)

    # Load and preprocess data
    data = pd.read_csv(data_path)
    existing_combinations = get_id_file_model_combinations(db_path)
    existing_df = pd.DataFrame(existing_combinations, columns=["id", "file", "model"])

    # Count existing rows for the current file and model
    rows_in_db_for_file_and_model = existing_df[
        (existing_df["file"] == data_path.name)
        & (existing_df["model"] == str(model_id))
    ].shape[0]

    # Create a set of existing combinations for faster lookup
    existing_set = set(
        (str(id), file, str(model)) for id, file, model in existing_combinations
    )

    # Add file and model columns to the data
    data["file"] = data_path.name
    data["model"] = str(model_id)

    # Filter out already processed rows
    data_filtered = data[
        ~data.apply(
            lambda row: (str(row["id"]), row["file"], str(row["model"]))
            in existing_set,
            axis=1,
        )
    ]

    # Limit the number of rows if max_rows is specified
    if max_rows is not None:
        data_filtered = data_filtered.head(max_rows)

    # Print processing information
    print(
        f"Number of rows in DB for file '{data_path.name}' and model '{model_id}': "
        f"{rows_in_db_for_file_and_model}"
    )
    print(f"Processing {len(data_filtered)} out of total {len(data)} rows")

    # Process rows in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _, row in data_filtered.iterrows():
            future = executor.submit(
                process,
                analyze_chain,
                so_chain,
                row["context"],
                row["question"],
                so_parser.get_format_instructions(),
                langfuse_handler,
            )
            futures.append((future, row))

        # Process results and insert into database
        for future, row in tqdm(futures, desc="Processing rows"):
            output, formatted_output = future.result()
            row_data = {
                "id": row["id"],
                "file": row["file"],
                "model": model_id,
                "title": row["title"],
                "reasoning": output,
                "structured_reasoning": formatted_output.reasoning,
                "relevance_label": formatted_output.relevance_label,
                "relevance_label_gt": row["relevance_label"],
            }
            insert_row(db_path, row_data)

    print("All rows processed and inserted into the database")


if __name__ == "__main__":
    # Initialize Aily settings
    AilySettings()

    # Load configuration and prompts
    base_path = Path("../")
    config = load_yaml(base_path / "config.yaml")
    prompts = load_yaml(base_path / "prompts.yaml")

    # Set up paths and model ID
    data_path = base_path / config["data_dir"] / config["data_file"]
    db_path = base_path / config["data_dir"] / config["results_file"]
    model_id = BedrockModelID.Claude35Sonnet20240620V1_0
    # model_id = OpenAIModelID.GPT4O

    # Run the main processing pipeline
    run(
        data_path=data_path,
        db_path=db_path,
        model_id=model_id,
        # max_rows=5,  # Uncomment to limit the number of rows processed
        max_workers=4,
    )
