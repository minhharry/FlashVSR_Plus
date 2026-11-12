"""Download pretrained checkpoints for FlashVSR+ from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_file(
    repo_id: str,
    filename: str,
    target_destination: Path,
) -> Path:
    target_destination.parent.mkdir(parents=True, exist_ok=True)
    if target_destination.exists() and target_destination.stat().st_size > 0:
        print(f"Already exists: {target_destination}")
        return target_destination

    print(f"Downloading {filename} from {repo_id}...")
    downloaded_cache_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
    )
    if downloaded_cache_path.resolve() != target_destination.resolve():
        import shutil
        shutil.copyfile(downloaded_cache_path, target_destination)
    return target_destination


def download_checkpoints(
    preset: str,
    ckpts_root: Path,
    include_vae: bool = False,
    only_decoder: bool = False,
) -> None:
    main_repo = "ruchangyao/FlashVSR_plus"
    v1_repo = "JunhaoZhuang/FlashVSR-v1.1"

    # Common streaming decoder needed by all presets
    download_file(
        repo_id=main_repo,
        filename="TCDecoder.ckpt",
        target_destination=ckpts_root / "TCDecoder.ckpt",
    )

    if only_decoder:
        print("TCDecoder downloaded successfully.")
        return

    if include_vae:
        download_file(
            repo_id=main_repo,
            filename="Wan2.1_VAE.pth",
            target_destination=ckpts_root / "Wan2.1_VAE.pth",
        )

    if preset in {"turbo", "all"}:
        print("Fetching Turbo model weights...")
        download_file(
            repo_id=main_repo,
            filename="turbo/dit.safetensors",
            target_destination=ckpts_root / "turbo" / "dit.safetensors",
        )
        download_file(
            repo_id=main_repo,
            filename="turbo/lq_proj_in.safetensors",
            target_destination=ckpts_root / "turbo" / "lq_proj_in.safetensors",
        )

    if preset in {"base", "all"}:
        print("Fetching Base model weights...")
        download_file(
            repo_id=main_repo,
            filename="base/dit.safetensors",
            target_destination=ckpts_root / "base" / "dit.safetensors",
        )
        download_file(
            repo_id=main_repo,
            filename="base/lq_proj_in.safetensors",
            target_destination=ckpts_root / "base" / "lq_proj_in.safetensors",
        )

    if preset in {"v1", "all"}:
        print("Fetching V1 model weights...")
        download_file(
            repo_id=v1_repo,
            filename="diffusion_pytorch_model_streaming_dmd.safetensors",
            target_destination=ckpts_root / "v1" / "dit.safetensors",
        )
        download_file(
            repo_id=v1_repo,
            filename="LQ_proj_in.ckpt",
            target_destination=ckpts_root / "v1" / "lq_proj_in.safetensors",
        )

    print("Checkpoints ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FlashVSR+ model checkpoints.")
    parser.add_argument(
        "--preset",
        choices=["turbo", "base", "v1", "all"],
        default="turbo",
        help="Model preset weights to download (default: turbo)",
    )
    parser.add_argument(
        "--ckpts-dir",
        type=Path,
        default=Path("ckpts"),
        help="Target directory for checkpoints (default: ckpts)",
    )
    parser.add_argument(
        "--include-vae",
        action="store_true",
        help="Also download optional Wan2.1_VAE.pth",
    )
    parser.add_argument(
        "--only-decoder",
        action="store_true",
        help="Only download common TCDecoder.ckpt",
    )
    arguments = parser.parse_args()
    download_checkpoints(
        arguments.preset,
        arguments.ckpts_dir,
        include_vae=arguments.include_vae,
        only_decoder=arguments.only_decoder,
    )


if __name__ == "__main__":
    main()

