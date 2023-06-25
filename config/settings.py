from pydantic import BaseSettings
from typing import List


class Settings(BaseSettings):
    api_name: str = "ChatGLM-6B API"
    api_version: str = "0.0.1"
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = True
    prefix: str = ""
    openapi_prefix: str = ""
    timeout_keep_alive: int = 120
    log_level: str = "debug"

    llm_model: str = "E:\\ai\\nlp\\llm\\THUDM\\chatglm-6b"
    torch_hub_dir: str = "E:\\ai\\nlp\\torch"



