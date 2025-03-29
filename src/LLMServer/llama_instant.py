import os
import json
import time
from typing import Generator
from llama_cpp import Llama
from enum import Enum
from abc import ABC

from LLMServer.base_model import BaseLLM
from tenacity import retry, stop_after_attempt, wait_exponential

STOP_AFTER_ATTEMPT = 10
WAIT_EXPONENTIAL_MIN = 4
WAIT_EXPONENTIAL_MAX = 30


def delayed_text_generator(text: str, delay: float = 0.2):
    tokens = text.split()
    for i in range(1, len(tokens) + 1):
        time.sleep(delay)
        yield " ".join(tokens[:i])


class ModelType(Enum):
    MISTRAL_7B = "mistral-7b"
    MINISTRAL_8B = "ministral-8b"
    MISTRAL_SMALL = "mistral-small"
    MISTRAL_SMALL3 = "mistral-small3"
    MIXTRAL_8_7B = "mixtral-8-7b"
    MIXTRAL_8_22B = "mixtral-8-22b"
    LLAMA_3_2_3B = "llama-3-2-3b"
    LLAMA_3_1_8B = "llama-3-1-8b"
    FALCON_7B = "falcon-7b"
    FALCON_40B = "falcon-40b"
    YI_6B = "yi-6b"
    YI_34B = "yi-34b"
    QWEN_7B = "qwen-7b"
    QWEN_14B = "qwen-14b"
    QWEN_72B = "qwen-72b"
    GEMMA_2B = "gemma-2b"
    GEMMA_7B = "gemma-7b"
    GEMMA2_9B = "gemma2-9b"
    GEMMA2_27B = "gemma2-27b"
    GROK_1 = "grok-1"
    TINY_LLAMA_1_1B = "tiny-llama-1.1b"
    PHI_2 = "phi-2"
    STABLE_LM_2_1_2B = "stable-lm-2-1.2b"
    NEURAL_7B_2 = "neural-7b-2"
    OPENCHAT_3_5 = "openchat-3.5"
    DEEPSEEK_R1_QWEN_1_5B = "deepseek_r1_qwen_1_5b"
    DEEPSEEK_R1_QWEN_7B = "deepseek_r1_qwen_7b"
    DEEPSEEK_R1_LLAMA_8B = "deepseek_r1_llama_8b"


class BaseQuantizedModel(ABC):
    def __init__(
        self,
        model_path: str,
        filename: str,
        n_ctx: int = 16384,
        n_gpu_layers: int = -1,
        # n_gpu_layers: int = 16,
        max_output_length: int = 1024,
        verbose: bool = False
    ):
        self.llm = Llama.from_pretrained(
            repo_id=model_path,
            filename=filename,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
        )
        self.inference_params = {
            "max_tokens": max_output_length,
            "temperature": 0,
            "top_p": 0.9,
        }

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN, max=WAIT_EXPONENTIAL_MAX),
    )
    def invoke(self, prompt: str) -> str:
        try:
            output = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **self.inference_params
            )
            return output["choices"][0]["message"]["content"]
        except Exception as e:
            raise ValueError(f"Incorrect Generation: {str(e)}")

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN, max=WAIT_EXPONENTIAL_MAX),
    )
    def stream_inference(self, prompt: str) -> Generator[str, None, None]:
        try:
            output = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **self.inference_params,
                stream=True
            )
            full_response = ""
            for chunk in output:
                if chunk["choices"][0]["delta"].get("content"):
                    full_response += chunk["choices"][0]["delta"]["content"]
                    yield chunk["choices"][0]["delta"]["content"]
        except Exception as e:
            raise ValueError(f"Incorrect Generation: {str(e)}")

class Mistral7BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_ctx: int = 16384
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Mistral7BModel:7B"

class MistralSmallModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "bartowski/Mistral-Small-Instruct-2409-GGUF",
        filename: str = "Mistral-Small-Instruct-2409-Q4_K_M.gguf",
        n_ctx: int = 16384
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "MistralSmallModel"

class Ministral8BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "bartowski/Ministral-8B-Instruct-2410-GGUF",
        filename: str = "Ministral-8B-Instruct-2410-Q4_K_M.gguf",
        n_ctx: int = 8192
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Ministral8BModel:8B"

class Mixtral8x7BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        filename: str = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx: int = 32768
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

class Mixtral8x22BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "bartowski/Mixtral-8x22B-v0.1-GGUF",
        filename: str = "Mixtral-8x22B-v0.1-IQ4_XS-00001-of-00005.gguf",
        n_ctx: int = 32768
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )


    def get_id(self):
        return "Mixtral8x7BModel:8x7B"

class Llama3_2_3BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF",
        filename: str = "*q8_0.gguf",
        n_ctx: int = 8192
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "LlamaModel:3.2-3B"

class Llama3_1_8BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
        filename: str = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        n_ctx: int = 8192
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "LlamaModel:3.1-8B"

class Falcon7BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/falcon-7b-instruct-GGUF",
        filename: str = "falcon-7b-instruct.Q4_K_M.gguf",
        n_ctx: int = 16384
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Falcon7BModel:7B"


class Falcon40BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/falcon-40b-instruct-GGUF",
        filename: str = "falcon-40b-instruct.Q4_K_M.gguf",
        n_ctx: int = 16384
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Falcon40BModel:40B"


# Yi model classes
class Yi6BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/Yi-6B-Chat-GGUF",
        filename: str = "yi-6b-chat.Q4_K_M.gguf",
        n_ctx: int = 16384
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Yi6BModel:6B"


class Yi34BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/Yi-34B-Chat-GGUF",
        filename: str = "yi-34b-chat.Q4_K_M.gguf",
        n_ctx: int = 32768
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Yi34BModel:34B"


# Qwen model classes
class Qwen7BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/Qwen-7B-Chat-GGUF",
        filename: str = "qwen-7b-chat.Q4_K_M.gguf",
        n_ctx: int = 16384
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Qwen7BModel:7B"


class Qwen14BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/Qwen-14B-Chat-GGUF",
        filename: str = "qwen-14b-chat.Q4_K_M.gguf",
        n_ctx: int = 16384
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Qwen14BModel:14B"

class Qwen72BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "bartowski/Qwen2.5-72B-0.6x-Instruct-GGUF",
        filename: str = "Qwen2.5-72B-0.6x-Instruct-IQ4_XS.gguf",
        n_ctx: int = 16384
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Qwen72BModel:72B"


# Gemma model classes
class Gemma2BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/Gemma-2b-it-GGUF",
        filename: str = "gemma-2b-it.Q4_K_M.gguf",
        n_ctx: int = 8192
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Gemma2BModel:2B"


class Gemma7BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/Gemma-7b-it-GGUF",
        filename: str = "gemma-7b-it.Q4_K_M.gguf",
        n_ctx: int = 8192
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Gemma7BModel:7B"


class Gemma2_9BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "bartowski/gemma-2-9b-it-GGUF",
        filename: str = "gemma-2-9b-it-Q4_K_M.gguf",
        # filename: str = "gemma-2-9b-it-Q6_K.gguf",
        n_ctx: int = 8192
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Gemma2_9BModel:9B"

class Gemma2_27BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "bartowski/gemma-2-27b-it-GGUF",
        filename: str = "gemma-2-27b-it-IQ4_XS.gguf",
        n_ctx: int = 8192
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Gemma2_27BModel:27B"

class DeepSeekR1Qwen7BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        filename: str = "DeepSeek-R1-Distill-Qwen-7B-Q4_K_L.gguf",
        n_ctx: int = 131072
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx,
            verbose=True,
        )

    def get_id(self):
        return "DeepSeekR1Qwen7BModel:7B"

class DeepSeekR1Qwen1_5BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        filename: str = "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        n_ctx: int = 131072
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx,
        )

    def get_id(self):
        return "DeepSeekR1Qwen1_5BModel:1.5B"
    
class DeepSeekR1Llama8BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        filename: str = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        n_ctx: int = 131072
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "DeepSeekR1Llama8BModel:8B"


# Grok model class (Note: This is a placeholder as Grok's availability might be limited)
class GrokModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/Grok-1-GGUF",  # Placeholder path
        filename: str = "grok-1.Q4_K_M.gguf",      # Placeholder filename
        n_ctx: int = 32768
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "GrokModel:1"


# New smaller model implementations
class TinyLlama1_1BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        n_ctx: int = 8192
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "TinyLlama1_1BModel:1.1B"


class Phi2Model(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/phi-2-GGUF",
        filename: str = "phi-2.Q4_K_M.gguf",
        n_ctx: int = 8192
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Phi2Model:2.7B"


class StableLM2_1_2BModel(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/stablelm-2-zephyr-1_2b-GGUF",
        filename: str = "stablelm-2-zephyr-1_2b.Q4_K_M.gguf",
        n_ctx: int = 8192
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "StableLM2Model:1.2B"


class Neural7B2Model(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/neural-chat-7b-v2-GGUF",
        filename: str = "neural-chat-7b-v2.Q4_K_M.gguf",
        n_ctx: int = 8192
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "Neural7B2Model:7B"


class OpenChat3_5Model(BaseQuantizedModel):
    def __init__(
        self,
        model_path: str = "TheBloke/openchat_3.5-GGUF",
        filename: str = "openchat_3.5.Q4_K_M.gguf",
        n_ctx: int = 8192
    ):
        super().__init__(
            model_path=model_path,
            filename=filename,
            n_ctx=n_ctx
        )

    def get_id(self):
        return "OpenChat3_5Model:3.5"

class ModelFactory:
    @staticmethod
    def create_model(model_type: ModelType, **kwargs) -> BaseQuantizedModel:
        model_map = {
            # Existing larger models
            ModelType.MISTRAL_7B: Mistral7BModel,
            ModelType.MINISTRAL_8B: Ministral8BModel,
            ModelType.MISTRAL_SMALL: MistralSmallModel,
            ModelType.MIXTRAL_8_7B: Mixtral8x7BModel,
            ModelType.MIXTRAL_8_22B: Mixtral8x22BModel,
            ModelType.LLAMA_3_2_3B: Llama3_2_3BModel,
            ModelType.LLAMA_3_1_8B: Llama3_1_8BModel,
            ModelType.FALCON_7B: Falcon7BModel,
            ModelType.FALCON_40B: Falcon40BModel,
            ModelType.YI_6B: Yi6BModel,
            ModelType.YI_34B: Yi34BModel,
            ModelType.QWEN_7B: Qwen7BModel,
            ModelType.QWEN_14B: Qwen14BModel,
            ModelType.QWEN_72B: Qwen72BModel,
            ModelType.GEMMA_2B: Gemma2BModel,
            ModelType.GEMMA_7B: Gemma7BModel,
            ModelType.GEMMA2_9B: Gemma2_9BModel,
            ModelType.GEMMA2_27B: Gemma2_27BModel,
            ModelType.GROK_1: GrokModel,
            ModelType.DEEPSEEK_R1_QWEN_1_5B: DeepSeekR1Qwen1_5BModel,
            ModelType.DEEPSEEK_R1_QWEN_7B: DeepSeekR1Qwen7BModel,
            ModelType.DEEPSEEK_R1_LLAMA_8B: DeepSeekR1Llama8BModel,
            
            # New smaller models
            ModelType.TINY_LLAMA_1_1B: TinyLlama1_1BModel,
            ModelType.PHI_2: Phi2Model,
            ModelType.STABLE_LM_2_1_2B: StableLM2_1_2BModel,
            ModelType.NEURAL_7B_2: Neural7B2Model,
            ModelType.OPENCHAT_3_5: OpenChat3_5Model,
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return model_map[model_type](**kwargs)
