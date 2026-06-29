"""SDX bundle catalog — text + vision model pairs, one per tier."""
from pathlib import Path

SDX_DIR = Path.home() / ".freeaiagent" / "sdx"

_HF = "https://huggingface.co"
_BK = f"{_HF}/bartowski"

SDX_CATALOG: dict[str, dict] = {
    "sdx-nano": {
        "display": "SDX Nano",
        "tagline": "Text + vision on any PC — ultra-light bundle (2.1 GB)",
        "kind": "sdx",
        "tier": "nano",
        "min_ram_gb": 4,
        "size_gb": 2.1,
        "token_budget": 4096,
        "description": "Smallest SDX bundle — text + basic vision on any machine.",
        "files": {
            "text": {
                "url": f"{_BK}/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
                "size_gb": 0.4,
                "sha256": None,
            },
            "vision": {
                "url": f"{_HF}/vikhyatk/moondream2/resolve/main/moondream2-text-model-f16.gguf",
                "size_gb": 1.7,
                "sha256": None,
                "mmproj": None,
            },
        },
    },
    "sdx-mini": {
        "display": "SDX Mini",
        "tagline": "Text + vision — small and sharp (2.5 GB)",
        "kind": "sdx",
        "tier": "mini",
        "min_ram_gb": 4,
        "size_gb": 2.5,
        "token_budget": 8192,
        "description": "Lightweight SDX with a stronger text model than Nano.",
        "files": {
            "text": {
                "url": f"{_BK}/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                "size_gb": 0.8,
                "sha256": None,
            },
            "vision": {
                "url": f"{_HF}/vikhyatk/moondream2/resolve/main/moondream2-text-model-f16.gguf",
                "size_gb": 1.7,
                "sha256": None,
                "mmproj": None,
            },
        },
    },
    "sdx-standard": {
        "display": "SDX Standard",
        "tagline": "The balanced all-rounder — chat, code, and images (4.7 GB)",
        "kind": "sdx",
        "tier": "standard",
        "min_ram_gb": 8,
        "size_gb": 4.7,
        "token_budget": 8192,
        "description": "Best default SDX — solid text and capable vision.",
        "files": {
            "text": {
                "url": f"{_BK}/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                "size_gb": 2.0,
                "sha256": None,
            },
            "vision": {
                "url": f"{_HF}/xtuner/llava-phi-3-mini-gguf/resolve/main/llava-phi-3-mini-int4.gguf",
                "size_gb": 2.4,
                "sha256": None,
                "mmproj": {
                    "url": f"{_HF}/xtuner/llava-phi-3-mini-gguf/resolve/main/llava-phi-3-mini-mmproj-f16.gguf",
                    "size_gb": 0.3,
                    "sha256": None,
                },
            },
        },
    },
    "sdx-plus": {
        "display": "SDX Plus",
        "tagline": "High-quality answers and strong vision (9.4 GB)",
        "kind": "sdx",
        "tier": "plus",
        "min_ram_gb": 16,
        "size_gb": 9.4,
        "token_budget": 16384,
        "description": "Strong reasoning text model paired with LLaVA 1.6 vision.",
        "files": {
            "text": {
                "url": f"{_BK}/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                "size_gb": 4.7,
                "sha256": None,
            },
            "vision": {
                "url": f"{_HF}/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-1.6-mistral-7b.Q4_K_M.gguf",
                "size_gb": 4.1,
                "sha256": None,
                "mmproj": {
                    "url": f"{_HF}/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf",
                    "size_gb": 0.6,
                    "sha256": None,
                },
            },
        },
    },
    "sdx-max": {
        "display": "SDX Max",
        "tagline": "Full-power text and vision — flagship machines only (13.4 GB)",
        "kind": "sdx",
        "tier": "max",
        "min_ram_gb": 24,
        "size_gb": 13.4,
        "token_budget": 32768,
        "description": "Maximum quality SDX — Qwen 14B text + LLaVA 1.6 vision.",
        "files": {
            "text": {
                "url": f"{_BK}/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
                "size_gb": 8.7,
                "sha256": None,
            },
            "vision": {
                "url": f"{_HF}/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-1.6-mistral-7b.Q4_K_M.gguf",
                "size_gb": 4.1,
                "sha256": None,
                "mmproj": {
                    "url": f"{_HF}/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf",
                    "size_gb": 0.6,
                    "sha256": None,
                },
            },
        },
    },
}


def is_installed(model_id: str) -> bool:
    return (SDX_DIR / model_id / "text.gguf").exists()


def model_paths(model_id: str) -> dict:
    """Return resolved file paths for a downloaded SDX bundle."""
    base = SDX_DIR / model_id
    mmproj = base / "mmproj.gguf"
    return {
        "text": str(base / "text.gguf"),
        "vision": str(base / "vision.gguf"),
        "mmproj": str(mmproj) if mmproj.exists() else None,
    }
