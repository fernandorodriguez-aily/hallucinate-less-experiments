from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from aily_ai_brain.common.enums import BedrockModelID, OpenAIModelID, OpenRouterModelID
from aily_ai_brain.llms.langchain import langchain_get_llm
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load and parse a YAML file into a Python dictionary.

    Args:
        path (Path): The path to the YAML file.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed YAML data.
    """
    with open(path, "r") as file:
        return yaml.safe_load(file)


def setup_basic_chain(
    model_id: Union[BedrockModelID, OpenAIModelID, OpenRouterModelID],
    prompt: str,
    langfuse_handler: BaseCallbackHandler,
    max_tokens: int,
    output_parser: Optional[BaseOutputParser] = None,
    sensitive_data: bool = True,
) -> Runnable:
    """
    Create a langhcain Runnable.

    Inputs:
        model_id: Union[BedrockModelID, OpenAIModelID] - Bedrock or OpenAI model id
        to use
        prompt: str - prompt template text to send to LLM
        langfuse_handler: CallbackHandler - langfuse handler for recording LLM
        inputs/outputs
    """
    prompt_template = ChatPromptTemplate.from_template(
        template=prompt,
    )
    llm = langchain_get_llm(
        langfuse_handler=langfuse_handler,
        sensitive_data=sensitive_data,
        model_id=model_id,
        max_tokens=max_tokens,
    )
    if output_parser:
        parser = output_parser
    else:
        parser = StrOutputParser()

    return prompt_template | llm | parser
