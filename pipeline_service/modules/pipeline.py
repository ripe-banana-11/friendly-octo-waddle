from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Literal, Optional

from PIL import Image
import pyspz
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import (
    GenerateRequest,
    GenerateResponse,
    TrellisParams,
    TrellisRequest,
    TrellisResult,
)
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.rmbg_manager import BackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.gs_generator.reconviagen_manager import ReconViaGenManager
from modules.utils import (
    secure_randint,
    set_random_seed,
    decode_image,
    to_png_base64,
    save_files,
)


class GenerationPipeline:
    def __init__(self, settings: Settings = settings):
        self.settings = settings

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings)
        self.rmbg = BackgroundRemovalService(settings)
        self.reconviagen = ReconViaGenManager(settings)

    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.rmbg.startup()
        try:
            await self.reconviagen.startup()
            if not self.reconviagen.is_ready():
                logger.warning("ReconViaGen failed to load, falling back to Trellis")
                raise RuntimeError("Only reconviagen is supported for now")
        except Exception as e:
            logger.error(f"ReconViaGen startup failed: {e}")
            raise RuntimeError("Only reconviagen is supported for now")

        self._clean_gpu_memory()

        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        if self.settings.use_reconviagen and self.reconviagen.is_ready():
            await self.reconviagen.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    # --- HÀM CỐT LÕI 1: CHUẨN BỊ ẢNH (CHỈ CHẠY 1 LẦN) ---
    async def prepare_input_images(
        self, image_bytes: bytes, seed: int = 42
    ) -> tuple[Image.Image, Image.Image]:
        """Chạy Qwen và RMBG để tạo view. Tách rời để dùng lại cho nhiều seed Trellis."""
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image = decode_image(image_base64)
        if seed < 0:
            seed = secure_randint(0, 10000)
        set_random_seed(seed)

        # 1. left view
        left_image_edited = self.qwen_edit.edit_image(
            prompt_image=image,
            seed=seed,
            prompt="Show this object in left three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        )

        # right view
        right_image_edited = self.qwen_edit.edit_image(
            prompt_image=image,
            seed=seed,
            prompt="Show this object in right three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        )

        # back view
        # back_image_edited = self.qwen_edit.edit_image(
        #     prompt_image=image,
        #     seed=seed,
        #     prompt="Show this object in back three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        # )

        # 2. Remove background
        left_image_without_background = self.rmbg.remove_background(left_image_edited)
        right_image_without_background = self.rmbg.remove_background(right_image_edited)
        # back_image_without_background = self.rmbg.remove_background(back_image_edited)
        original_image_without_background = self.rmbg.remove_background(image)

        return [
            left_image_without_background,
            right_image_without_background,
            # back_image_without_background,
            original_image_without_background,
        ]

    # --- HÀM CỐT LÕI 2: CHẠY TRELLIS (CHẠY NHIỀU LẦN VỚI SEED KHÁC NHAU) ---
    async def generate_trellis_only(
        self,
        processed_images: list[Image.Image],
        seed: int,
        mode: Literal[
            "single", "multi_multi", "multi_sto", "multi_with_voxel_count"
        ] = "multi_with_voxel_count",
    ) -> bytes:
        """Chỉ chạy tạo 3D từ ảnh đã xử lý."""
        trellis_params = TrellisParams.from_settings(self.settings)
        set_random_seed(seed)

        trellis_result = self.trellis.generate(
            TrellisRequest(
                images=processed_images,
                seed=seed,
                params=trellis_params,
            ),
            mode=mode,
        )

        if not trellis_result or not trellis_result.ply_file:
            raise ValueError("Trellis generation failed")

        return trellis_result.ply_file

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return PLY as bytes.

        Args:
            image_bytes: Raw image bytes from uploaded file

        Returns:
            PLY file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Create request
        request = GenerateRequest(
            prompt_image=image_base64, prompt_type="image", seed=seed
        )

        # Generate
        response = await self.generate_gs(request)

        # Return binary PLY
        if not response.ply_file_base64:
            raise ValueError("PLY generation failed")

        return response.ply_file_base64  # bytes

    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute full generation pipeline.

        Args:
            request: Generation request with prompt and settings

        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"New generation request")

        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # 1. Edit the image using Qwen Edit
        image_edited = self.qwen_edit.edit_image(
            prompt_image=image,
            seed=request.seed,
            prompt="Show this object in left three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        )

        # 2. Remove background
        image_without_background = self.rmbg.remove_background(image_edited)

        # add another view of the image
        image_edited_2 = self.qwen_edit.edit_image(
            prompt_image=image,
            seed=request.seed,
            prompt="Show this object in right three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        )
        image_without_background_2 = self.rmbg.remove_background(image_edited_2)

        # add another view of the image
        image_edited_3 = self.qwen_edit.edit_image(
            prompt_image=image,
            seed=request.seed,
            prompt="Show this object in back view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        )
        image_without_background_3 = self.rmbg.remove_background(image_edited_3)

        # save to debug
        # image_edited.save("image_edited.png")
        # image_edited_2.save("image_edited_2.png")
        # image_without_background.save("image_without_background.png")
        # image_without_background_2.save("image_without_background_2.png")

        trellis_result: Optional[TrellisResult] = None

        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params

        # 3. Generate the 3D model
        # Choose between Trellis and ReconViaGen based on settings
        use_reconviagen = getattr(self.settings, 'use_reconviagen', False)
        
        if use_reconviagen and self.reconviagen.is_ready():
            logger.info("Using ReconViaGen for multi-view 3D generation")
            
            # Extract parameters for ReconViaGen
            # Use TrellisParams defaults if not provided
            if trellis_params:
                ss_guidance_strength = trellis_params.sparse_structure_cfg_strength
                ss_sampling_steps = trellis_params.sparse_structure_steps
                slat_guidance_strength = trellis_params.slat_cfg_strength
                slat_sampling_steps = trellis_params.slat_steps
            else:
                # Use defaults from settings
                ss_guidance_strength = self.settings.trellis_sparse_structure_cfg_strength
                ss_sampling_steps = self.settings.trellis_sparse_structure_steps
                slat_guidance_strength = self.settings.trellis_slat_cfg_strength
                slat_sampling_steps = self.settings.trellis_slat_steps
            
            multiimage_algo = getattr(self.settings, 'reconviagen_multiimage_algo', 'multidiffusion')
            
            trellis_result = self.reconviagen.generate_multiview(
                images=[image_without_background, image_without_background_2, image_without_background_3],
                seed=request.seed,
                ss_guidance_strength=ss_guidance_strength,
                ss_sampling_steps=ss_sampling_steps,
                slat_guidance_strength=slat_guidance_strength,
                slat_sampling_steps=slat_sampling_steps,
                multiimage_algo=multiimage_algo
            )
        else:
            logger.info("Using standard Trellis for multi-view 3D generation")
            trellis_result = self.trellis.generate(
                TrellisRequest(
                    images=[image_without_background, image_without_background_2, image_without_background_3],
                    seed=request.seed,
                    params=trellis_params,
                )
            )

        # Save generated files
        if self.settings.save_generated_files:
            save_files(
                trellis_result, 
                image, 
                image_edited, 
                image_without_background,
                image_edited_2,
                image_without_background_2,
                image_edited_3,
                image_without_background_3
            )

        # Convert to PNG base64 for response (only if needed)
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited)
            image_without_background_base64 = to_png_base64(image_without_background)

        t2 = time.time()
        generation_time = t2 - t1

        logger.info(f"Total generation time: {generation_time} seconds")
        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result.ply_file if trellis_result else None,
            image_edited_file_base64=image_edited_base64
            if self.settings.send_generated_files
            else None,
            image_without_background_file_base64=image_without_background_base64
            if self.settings.send_generated_files
            else None,
        )
        return response
