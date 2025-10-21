"""DeepSeek integration helper.

Loads the DeepSeek model locally (if dependencies available) and exposes an
infer_image_to_markdown helper. This is optional and will only be used if the
environment or configuration points to a DeepSeek model.
"""
import os
import tempfile
import logging
from typing import Optional

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from PIL import Image
except Exception:
    torch = None
    AutoModel = None
    AutoTokenizer = None

logger = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load(self):
        if AutoModel is None:
            raise RuntimeError("transformers/torch not available for DeepSeek")
        if self.model is None:
            logger.info("Loading DeepSeek model %s on device %s", self.model_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                _attn_implementation="flash_attention_2",
                trust_remote_code=True,
                use_safetensors=True,
            )
            # place on device, set bfloat16 if supported
            try:
                if self.device.startswith("cuda"):
                    self.model = self.model.to(torch.bfloat16).cuda()
                else:
                    self.model = self.model.to(torch.bfloat16)
            except Exception:
                # fallback to fp32 CPU
                logger.exception("Failed to cast model to bfloat16; falling back to CPU fp32")
                self.model = self.model.to("cpu")

    def infer_image_to_markdown(self, image_path: str, prompt: Optional[str] = None, output_path: Optional[str] = None, base_size: int = 1024, image_size: int = 640, crop_mode: bool = True):
        """Run DeepSeek inference on an image and return the markdown output.

        This function tries to load the model lazily and will save results to
        `output_path` if provided.
        """
        if self.model is None:
            self.load()

        prompt = prompt or "<image>\n<|grounding|>Convert the document to markdown. "
        out_path = output_path or tempfile.mkdtemp()
        try:
            res = self.model.infer(self.tokenizer, prompt=prompt, image_file=image_path, output_path=out_path, base_size=base_size, image_size=image_size, crop_mode=crop_mode, save_results=False)
            # model.infer may return textual output or write files; try to return string
            if isinstance(res, dict):
                # For some models, the main text is returned in 'output' or 'text'
                return res.get("output") or res.get("text") or str(res)
            if isinstance(res, str):
                return res
            # try to find a markdown result in the output path
            for root, _, files in os.walk(out_path):
                for f in files:
                    if f.lower().endswith(".md"):
                        with open(os.path.join(root, f), "r", encoding="utf-8") as fh:
                            return fh.read()
            return str(res)
        except Exception as exc:
            logger.exception("DeepSeek inference failed: %s", exc)
            raise


def get_client(model_name: Optional[str] = None) -> DeepSeekClient:
    return DeepSeekClient(model_name or os.environ.get("HUGGINGFACE_MODEL_SUMMARY", "deepseek-ai/DeepSeek-OCR"))
