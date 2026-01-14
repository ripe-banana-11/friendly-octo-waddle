"""
ReconViaGen Manager
Handles ReconViaGen pipeline for multi-view 3D generation
"""

from __future__ import annotations

import time
import gc
from typing import Optional, List
import torch
from PIL import Image

from config import Settings
from logger_config import logger
from schemas import TrellisResult, TrellisRequest, TrellisParams
from libs.trellis.pipelines.trellis_image_to_3d import TrellisVGGTTo3DPipeline

import io
class ReconViaGenManager:
    """Manager for ReconViaGen multi-view 3D generation"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: Optional[TrellisVGGTTo3DPipeline] = None
        self.gpu = getattr(settings, 'reconviagen_gpu', 0)
        self.default_params = TrellisParams.from_settings(settings)
        
    async def startup(self) -> None:
        """Initialize ReconViaGen pipeline"""
        logger.info("Loading ReconViaGen VGGT pipeline...")
        
        try:
            model_path = getattr(self.settings, 'reconviagen_model_id', "Stable-X/trellis-vggt-v0-1")
            self.pipeline = TrellisVGGTTo3DPipeline.from_pretrained(model_path)
            self.pipeline.cuda()
            # Explicitly move VGGT_model and birefnet_model to CUDA
            # (they are not in self.models dict, so cuda() doesn't move them)
            if hasattr(self.pipeline, 'VGGT_model') and self.pipeline.VGGT_model is not None:
                self.pipeline.VGGT_model.cuda()
            if hasattr(self.pipeline, 'birefnet_model') and self.pipeline.birefnet_model is not None:
                self.pipeline.birefnet_model.cuda()
            self.pipeline.is_loaded = True
                
            logger.success("ReconViaGen VGGT pipeline ready.")
            
        except Exception as e:
            logger.error(f"Failed to initialize ReconViaGen VGGT pipeline: {e}")
            self.pipeline = None
    
    async def shutdown(self) -> None:
        """Shutdown ReconViaGen pipeline"""
        if self.pipeline:
            self.pipeline.unload_pipeline()
            self.pipeline = None
        logger.info("ReconViaGen VGGT pipeline closed.")
    
    def is_ready(self) -> bool:
        """Check if pipeline is ready"""
        return self.pipeline is not None and self.pipeline.is_loaded

    def generate_shapes_only(
        self,
        trellis_request: TrellisRequest,
        num_samples: int = 5,
    ) -> tuple[list[tuple[torch.Tensor, int]], torch.Tensor, List]:
        """
        Generate only sparse structures (shapes) for multiple candidates.
        Returns list of (coords, voxel_count) tuples.
        
        Args:
            trellis_request: Request with images and parameters
            num_samples: Number of shapes to generate
            
        Returns:
            List of (coords tensor, voxel count) tuples
        """
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        images_rgb = [image.convert("RGB") for image in trellis_request.images]
        logger.info(f"🎲 Generating {num_samples} shape candidates (sparse structures only)...")

        params = trellis_request.params
        
        torch.manual_seed(trellis_request.seed)
        start = time.time()
        
        try:
            aggregated_tokens_list, _ = self.pipeline.vggt_feat(images_rgb)
            b, n, _, _ = aggregated_tokens_list[0].shape
            image_cond = self.pipeline.encode_image(images_rgb).reshape(b, n, -1, 1024)
            
            ss_flow_model = self.pipeline.models['sparse_structure_flow_model']
            ss_cond = self.pipeline.get_ss_cond(image_cond[:, :, 5:], aggregated_tokens_list, num_samples)
            # Sample structured latent
            ss_sampler_params = {"steps": params.sparse_structure_steps, "cfg_strength": params.sparse_structure_cfg_strength}
            reso = ss_flow_model.resolution
            ss_noise = torch.randn(num_samples, ss_flow_model.in_channels, reso, reso, reso).to(self.pipeline.device)
            ss_latent = self.pipeline.sparse_structure_sampler.sample(
                ss_flow_model,
                ss_noise,
                **ss_cond,
                **ss_sampler_params,
                verbose=True
            ).samples

            decoder = self.pipeline.models['sparse_structure_decoder']
            coords = torch.argwhere(decoder(ss_latent)>0)[:, [0, 2, 3, 4]].int()
            
            # Split coords by sample and count voxels
            results = []
            for sample_idx in range(num_samples):
                sample_coords = coords[coords[:, 0] == sample_idx]
                voxel_count = len(sample_coords)
                results.append((sample_coords, voxel_count))
                logger.info(f"   Shape {sample_idx+1}: {voxel_count} voxels")
            
            generation_time = time.time() - start
            logger.success(f"✅ Generated {num_samples} shapes in {generation_time:.2f}s")
            
            return results, image_cond, aggregated_tokens_list
            
        except Exception as e:
            logger.error(f"Shape generation failed: {e}")
            raise

    def generate_textures_for_shapes(
        self,
        trellis_request: TrellisRequest,
        shapes: list[tuple[torch.Tensor, int]],
        num_samples,
        image_cond,
        aggregated_tokens_list,
    ) -> list[TrellisResult]:
        """
        Generate textures for multiple shapes in ONE batched pass.
        
        Args:
            trellis_request: Request with images and parameters
            shapes: List of (coords, voxel_count) tuples
            
        Returns:
            List of TrellisResult with PLY files
        """
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        images_rgb = [image.convert("RGB") for image in trellis_request.images]
        logger.info(f"🎨 Batch generating {len(shapes)} textures (one per shape)...")
        for idx, (_, voxel_count) in enumerate(shapes):
            logger.info(f"   Shape {idx+1}: {voxel_count} voxels")

        params = self.default_params.overrided(trellis_request.params)
        cond = self.pipeline.get_cond(images_rgb)
        cond["neg_cond"] = cond["neg_cond"][:1]
        
        # Use max voxel count to determine steps (conservative approach)
        max_voxel_count = max(voxel_count for _, voxel_count in shapes)
        base_slat_steps = params.slat_steps
        voxel_threshold = 25000
        
        if max_voxel_count > voxel_threshold:
            adjusted_slat_steps = base_slat_steps
            logger.info(f"Max voxel count {max_voxel_count} > {voxel_threshold}: Using standard texture steps ({adjusted_slat_steps})")
        else:
            adjusted_slat_steps = int(base_slat_steps * 1.5)
            logger.info(f"Max voxel count {max_voxel_count} <= {voxel_threshold}: Using increased texture steps ({adjusted_slat_steps})")
        
        # Combine all coords into one batch tensor
        # Re-index sample_idx for each shape
        combined_coords_list = []
        for new_sample_idx, (coords, _) in enumerate(shapes):
            # Update sample index (first column) to new batch index
            batch_coords = coords.clone()
            batch_coords[:, 0] = new_sample_idx
            combined_coords_list.append(batch_coords)
        
        combined_coords = torch.cat(combined_coords_list, dim=0)
        logger.info(f"📦 Combined {len(shapes)} shapes into batched coords: {combined_coords.shape}")
        
        start = time.time()
        
        try:
            # Set seed for batched generation
            torch.manual_seed(trellis_request.seed + 100)

            slat_cond = self.pipeline.get_slat_cond(image_cond, aggregated_tokens_list, num_samples)
            # Handle multi-image conditioning by using inject_sampler_multi_image
            num_images = len(images_rgb)
            slat_steps = {"steps": adjusted_slat_steps, "cfg_strength": params.slat_cfg_strength}
            with self.pipeline.inject_sampler_multi_image('slat_sampler', num_images, slat_steps, mode="multidiffusion"):
                slat = self.pipeline.sample_slat(slat_cond, combined_coords, {"steps": adjusted_slat_steps, "cfg_strength": params.slat_cfg_strength})
            outputs = self.pipeline.decode_slat(slat, formats=["gaussian"])
            gaussians = outputs["gaussian"]  # List of Gaussian models
            
            # Convert each to PLY
            results = []
            for i, gaussian in enumerate(gaussians):
                buffer = io.BytesIO()
                gaussian.save_ply(buffer)
                buffer.seek(0)
                result = TrellisResult(ply_file=buffer.getvalue())
                buffer.close()
                results.append(result)
            
            generation_time = time.time() - start
            logger.success(f"✅ Batch generated {len(shapes)} textures in {generation_time:.2f}s ({generation_time/len(shapes):.2f}s per texture)")
            
            return results
            
        except Exception as e:
            logger.error(f"Texture generation failed: {e}")
            raise