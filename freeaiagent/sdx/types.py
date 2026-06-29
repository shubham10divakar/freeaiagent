from dataclasses import dataclass
from typing import Optional


@dataclass
class SDXMessage:
    role: str  # "user" | "assistant" | "system"
    content: str
    image_description: Optional[str] = None


@dataclass
class SDXBundle:
    id: str
    display: str
    tagline: str
    tier: str
    min_ram_gb: int
    token_budget: int
    text_url: str
    text_size_gb: float
    vision_url: str
    vision_size_gb: float
    mmproj_url: Optional[str] = None
    mmproj_size_gb: Optional[float] = None
