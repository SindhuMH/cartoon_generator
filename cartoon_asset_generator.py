"""
cartoon_asset_generator.py

Reusable Stable Diffusion XL + IP-Adapter asset generator.

This converts the core workflow from the supplied notebook into an importable
Python module. It can:
  1. Load SDXL + IP-Adapter
  2. Generate character/place variants
  3. Save a manifest.json
  4. Set a canonical variant
  5. Be imported and called from another Python program

Example:
    from cartoon_asset_generator import CartoonAssetGenerator, DEFAULT_CONFIG

    generator = CartoonAssetGenerator(
        output_root="/content/drive/MyDrive/panchatantra_assets",
        style_reference="/content/drive/MyDrive/panchatantra_assets/style_reference.png",
    )

    generator.generate_story_assets(DEFAULT_CONFIG, variants_per_subject=4)
"""

from pathlib import Path
import json
from typing import Dict, Optional, Union

import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image
from transformers import CLIPVisionModelWithProjection


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "story_id": "monkey_and_crocodile",
    "style": {
        "art_style": "detailed storybook illustration, forest setting",
        "negative_prompt": (
            "photorealistic, more than one character if its a character image, "
            "scary looking character if its a character image, 3d render, "
            "flat vector, watermark, text, signature, blurry, extra limbs, "
            "deformed, low quality"
        ),
        "character_size": [1024, 1024],
        "place_size": [1344, 768],
    },
    "characters": [
        {
            "id": "monkey",
            "name": "Chintu the Monkey",
            "description": (
                "a friendly young brown monkey with big expressive round eyes, "
                "wearing a small red vest, long curled tail, cheerful open-mouth "
                "smile, standing pose, full body, plain neutral background"
            ),
        },
        {
            "id": "crocodile",
            "name": "Kalu the Crocodile",
            "description": (
                "a slightly chubby green crocodile with rounded snout, small "
                "friendly eyes, textured back scales stylized as soft bumps, "
                "short stubby legs, standing pose, full body, plain neutral background"
            ),
        },
    ],
    "places": [
        {
            "id": "riverbank",
            "name": "River Bank",
            "description": (
                "a lush green riverbank at soft morning light, a large fruit "
                "tree with overhanging branches on one side, calm blue river "
                "with gentle ripples, distant hills, wide establishing shot, "
                "no characters"
            ),
        },
        {
            "id": "river_middle",
            "name": "Middle of the River",
            "description": (
                "view from the surface of a calm wide river, gentle ripples, "
                "soft morning sky reflected in the water, a few lily pads, "
                "wide shot, no characters"
            ),
        },
    ],
}


class CartoonAssetGenerator:
    """
    Generate consistent cartoon character and location assets using
    SDXL + IP-Adapter.

    The class is intentionally independent of the story parser and future
    video generator. That lets us later plug it into:

        story -> JSON -> assets -> scenes -> video
    """

    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    IP_ADAPTER_REPO = "h94/IP-Adapter"
    IP_ADAPTER_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"

    def __init__(
        self,
        output_root: Union[str, Path],
        style_reference: Union[str, Path],
        ip_adapter_scale: float = 0.45,
        device: Optional[str] = None,
        seed: int = 42,
        num_inference_steps: int = 25,
        guidance_scale: float = 6.0,
    ):
        self.output_root = Path(output_root)
        self.style_reference_path = Path(style_reference)

        self.output_root.mkdir(parents=True, exist_ok=True)

        self.ip_adapter_scale = ip_adapter_scale
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.pipe = None
        self.style_reference = None

        print(f"Device: {self.device}")
        print(f"Output root: {self.output_root}")
        print(f"Style reference: {self.style_reference_path}")

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def load(self):
        """Load SDXL and IP-Adapter."""

        if not self.style_reference_path.exists():
            raise FileNotFoundError(
                f"Style reference not found: {self.style_reference_path}"
            )

        self.style_reference = Image.open(
            self.style_reference_path
        ).convert("RGB")

        print(f"Reference image size: {self.style_reference.size}")

        # CPU can run in float32. CUDA uses float16 for memory efficiency.
        if self.device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

        print("Loading CLIP vision encoder...")

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.IP_ADAPTER_REPO,
            subfolder="models/image_encoder",
            torch_dtype=dtype,
        )

        print("Loading SDXL pipeline...")

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.MODEL_ID,
            image_encoder=image_encoder,
            torch_dtype=dtype,
            variant="fp16" if self.device == "cuda" else None,
        )

        print("Loading IP-Adapter...")

        self.pipe.load_ip_adapter(
            self.IP_ADAPTER_REPO,
            subfolder="sdxl_models",
            weight_name=self.IP_ADAPTER_WEIGHT,
        )

        self.pipe.set_ip_adapter_scale(self.ip_adapter_scale)

        # The original notebook used enable_model_cpu_offload().
        # That requires Accelerate and an appropriate accelerator setup.
        # For portability we explicitly move the pipeline instead.
        if self.device == "cuda":
            self.pipe = self.pipe.to("cuda")
        else:
            self.pipe = self.pipe.to("cpu")

        print("Model loaded successfully.")

        return self

    # -----------------------------------------------------------------------
    # Prompt construction
    # -----------------------------------------------------------------------

    @staticmethod
    def build_character_prompt(
        subject_description: str,
        style: Dict,
    ) -> str:
        return (
            f"Generate an image with only one character: "
            f"{subject_description}, {style['art_style']}"
        )

    @staticmethod
    def build_place_prompt(
        subject_description: str,
        style: Dict,
    ) -> str:
        return (
            f"Generate an image of: "
            f"{subject_description}, {style['art_style']}"
        )

    # -----------------------------------------------------------------------
    # Image generation
    # -----------------------------------------------------------------------

    def _generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: Optional[int] = None,
    ) -> Image.Image:

        if self.pipe is None:
            raise RuntimeError(
                "Model is not loaded. Call generator.load() first."
            )

        if seed is None:
            seed = self.seed

        # Keep generator on the same device as the pipeline.
        generator = torch.Generator(
            device=self.device
        ).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=self.style_reference,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return result

    # -----------------------------------------------------------------------
    # Generate one subject
    # -----------------------------------------------------------------------

    def generate_variants(
        self,
        subject_id: str,
        subject_description: str,
        style: Dict,
        size,
        out_dir: Union[str, Path],
        num_variants: int,
        manifest: Dict,
        category: str,
    ):
        """Generate several variants of one character or place."""

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if category == "characters":
            prompt = self.build_character_prompt(
                subject_description,
                style,
            )
        elif category == "places":
            prompt = self.build_place_prompt(
                subject_description,
                style,
            )
        else:
            raise ValueError(
                "category must be 'characters' or 'places'"
            )

        negative_prompt = style.get("negative_prompt", "")

        manifest_entry = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "ip_adapter_scale": self.ip_adapter_scale,
            "variants": [],
        }

        width, height = size

        for i in range(1, num_variants + 1):

            # Slightly different seed per variant while remaining reproducible.
            variant_seed = self.seed + i - 1

            print(
                f"[{category}:{subject_id}] "
                f"generating variant {i}/{num_variants} "
                f"(seed={variant_seed})..."
            )

            image = self._generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=variant_seed,
            )

            variant_path = out_dir / f"variant_{i}.png"
            image.save(variant_path)

            manifest_entry["variants"].append(
                {
                    "variant": i,
                    "seed": variant_seed,
                    "path": str(variant_path),
                }
            )

            print(f"Saved: {variant_path}")

        manifest[category][subject_id] = manifest_entry

    # -----------------------------------------------------------------------
    # Generate complete story assets
    # -----------------------------------------------------------------------

    def generate_story_assets(
        self,
        config: Dict,
        variants_per_subject: int = 4,
    ) -> Dict:
        """
        Generate all characters and places from the story configuration.

        Returns:
            manifest dictionary
        """

        story_id = config["story_id"]
        style = config["style"]

        base_dir = self.output_root / story_id
        base_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "story_id": story_id,
            "style": style,
            "characters": {},
            "places": {},
        }

        # Characters
        for char in config.get("characters", []):

            self.generate_variants(
                subject_id=char["id"],
                subject_description=char["description"],
                style=style,
                size=style["character_size"],
                out_dir=base_dir / "characters" / char["id"],
                num_variants=variants_per_subject,
                manifest=manifest,
                category="characters",
            )

        # Places
        for place in config.get("places", []):

            self.generate_variants(
                subject_id=place["id"],
                subject_description=place["description"],
                style=style,
                size=style["place_size"],
                out_dir=base_dir / "places" / place["id"],
                num_variants=variants_per_subject,
                manifest=manifest,
                category="places",
            )

        manifest_path = base_dir / "manifest.json"

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print()
        print("=" * 60)
        print("ASSET GENERATION COMPLETE")
        print("=" * 60)
        print(f"Story:    {story_id}")
        print(f"Manifest: {manifest_path}")
        print("=" * 60)

        return manifest

    # -----------------------------------------------------------------------
    # Canonical image
    # -----------------------------------------------------------------------

    def set_canonical(
        self,
        manifest: Dict,
        category: str,
        subject_id: str,
        variant_number: int,
    ) -> Path:
        """
        Set one generated variant as the canonical reference image.

        Example:
            generator.set_canonical(
                manifest,
                "characters",
                "monkey",
                2
            )
        """

        entry = manifest[category][subject_id]
        variants = entry["variants"]

        if not 1 <= variant_number <= len(variants):
            raise ValueError(
                f"variant_number must be between 1 and {len(variants)}"
            )

        src = Path(variants[variant_number - 1]["path"])
        dst = src.parent / "canonical.png"

        dst.write_bytes(src.read_bytes())

        # Store canonical information in manifest.
        entry["canonical"] = {
            "variant": variant_number,
            "path": str(dst),
        }

        print(f"Canonical set: {dst}")

        return dst

    # -----------------------------------------------------------------------
    # Save updated manifest
    # -----------------------------------------------------------------------

    @staticmethod
    def save_manifest(manifest: Dict, path: Union[str, Path]):
        path = Path(path)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"Manifest saved: {path}")


# ---------------------------------------------------------------------------
# Simple convenience function
# ---------------------------------------------------------------------------

def generate_assets(
    config: Dict,
    output_root: Union[str, Path],
    style_reference: Union[str, Path],
    variants_per_subject: int = 4,
    seed: int = 42,
):
    """
    Convenience wrapper for users who don't need the class directly.

    Example:
        from cartoon_asset_generator import generate_assets

        manifest = generate_assets(
            config,
            output_root="./assets",
            style_reference="./style_reference.png",
        )
    """

    generator = CartoonAssetGenerator(
        output_root=output_root,
        style_reference=style_reference,
        seed=seed,
    )

    generator.load()

    return generator.generate_story_assets(
        config=config,
        variants_per_subject=variants_per_subject,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate cartoon character and location assets."
    )

    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where generated assets will be stored.",
    )

    parser.add_argument(
        "--style-reference",
        required=True,
        help="Path to the style reference image.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON configuration file.",
    )

    parser.add_argument(
        "--variants",
        type=int,
        default=4,
        help="Number of variants per character/place.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG

    generator = CartoonAssetGenerator(
        output_root=args.output_root,
        style_reference=args.style_reference,
        seed=args.seed,
    )

    generator.load()

    generator.generate_story_assets(
        config=config,
        variants_per_subject=args.variants,
    )
