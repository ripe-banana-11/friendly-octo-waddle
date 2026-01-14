from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from contextlib import contextmanager
from PIL import Image
import os
import sys

from .base import Pipeline
from . import samplers
from ..modules import sparse as sp

# Add the wheels directory to the path for VGGT
wheels_path = os.path.join(os.path.dirname(__file__), "..", "..", "wheels", "vggt")
if wheels_path not in sys.path:
    sys.path.insert(0, wheels_path)
from vggt.models.vggt import VGGT
from transformers import AutoModelForImageSegmentation


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    default_image_resolution = 518

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(
            TrellisImageTo3DPipeline, TrellisImageTo3DPipeline
        ).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(
            samplers, args["sparse_structure_sampler"]["name"]
        )(**args["sparse_structure_sampler"]["args"])
        new_pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"][
            "params"
        ]

        new_pipeline.slat_sampler = getattr(samplers, args["slat_sampler"]["name"])(
            **args["slat_sampler"]["args"]
        )
        new_pipeline.slat_sampler_params = args["slat_sampler"]["params"]

        new_pipeline.slat_normalization = args["slat_normalization"]

        new_pipeline._init_image_cond_model(args["image_cond_model"])

        return new_pipeline

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
        dinov2_model.eval()
        self.models["image_cond_model"] = dinov2_model
        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = (
            np.min(bbox[:, 1]),
            np.min(bbox[:, 0]),
            np.max(bbox[:, 1]),
            np.max(bbox[:, 0]),
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2,
        )
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(
        self, image: Union[torch.Tensor, list[Image.Image]]
    ) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), (
                "Image list should be list of PIL images"
            )
            image = [i.resize((518, 518), Image.Resampling.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models["image_cond_model"](image, is_training=True)["x_prenorm"]
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens

    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            "cond": cond,
            "neg_cond": neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(
            self.device
        )
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=True
        ).samples

        # Decode occupancy latent
        decoder = self.models["sparse_structure_decoder"]
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ["gaussian"],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if "gaussian" in formats:
            ret["gaussian"] = self.models["slat_decoder_gs"](slat)

        return ret

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models["slat_flow_model"]
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["gaussian"],
        preprocess_image: bool = True,
        *,
        num_oversamples: int = 1,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        num_oversamples = max(num_samples, num_oversamples)
        coords = self.sample_sparse_structure(
            cond, num_oversamples, sparse_structure_sampler_params
        )
        coords = (
            coords
            if num_oversamples <= num_samples
            else self.select_coords(coords, num_samples)
        )
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    def select_coords(self, coords, num_samples):
        """
        Select n smallest sparse structures in terms of number of voxels
        """
        counts = coords[:, 0].unique(return_counts=True)[-1]
        selected_coords = sorted(
            coords[:, 1:].split(tuple(counts.tolist())), key=lambda x: len(x)
        )[:num_samples]
        sizes = torch.tensor(tuple(len(coo) for coo in selected_coords))
        selected_coords = torch.cat(selected_coords, dim=0)
        indices = (
            torch.arange(num_samples)
            .repeat_interleave(sizes)
            .unsqueeze(-1)
            .to(selected_coords.device, selected_coords.dtype)
        )
        selected_coords = torch.cat((indices, selected_coords), dim=1)
        return selected_coords

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal["stochastic", "multidiffusion"] = "multidiffusion",
    ):
        """
        Inject a sampler with multiple images as condition.

        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f"_old_inference_model", sampler._inference_model)

        if mode == "stochastic":
            if num_images > num_steps:
                print(
                    f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m"
                )

            cond_indices = (np.arange(num_steps) % num_images).tolist()

            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx : cond_idx + 1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

        elif mode == "multidiffusion":
            from .samplers import FlowEulerSampler

            def _new_inference_model(
                self,
                model,
                x_t,
                t,
                cond,
                neg_cond,
                cfg_strength,
                cfg_interval,
                **kwargs,
            ):
                # Handle both cond and neg_cond as lists
                # get_slat_cond returns lists when there are multiple images
                if isinstance(cond, list):
                    cond_list = cond
                else:
                    cond_list = [cond] if cond is not None else []
                
                if isinstance(neg_cond, list):
                    neg_cond_list = neg_cond
                else:
                    neg_cond_list = [neg_cond] if neg_cond is not None else []
                
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond_list)):
                        preds.append(
                            FlowEulerSampler._inference_model(
                                self, model, x_t, t, cond_list[i], **kwargs
                            )
                        )
                    pred = sum(preds) / len(preds)
                    # Use first neg_cond (should be zeros anyway)
                    neg_pred_val = neg_cond_list[0] if len(neg_cond_list) > 0 else (neg_cond if not isinstance(neg_cond, list) else None)
                    if neg_pred_val is not None:
                        neg_pred = FlowEulerSampler._inference_model(
                            self, model, x_t, t, neg_pred_val, **kwargs
                        )
                    else:
                        neg_pred = torch.zeros_like(pred)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond_list)):
                        preds.append(
                            FlowEulerSampler._inference_model(
                                self, model, x_t, t, cond_list[i], **kwargs
                            )
                        )
                    pred = sum(preds) / len(preds)
                    return pred

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f"_old_inference_model")

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["gaussian"],
        preprocess_image: bool = True,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
        num_oversamples: int = 1,
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond["neg_cond"] = cond["neg_cond"][:1]
        torch.manual_seed(seed)
        num_oversamples = max(num_samples, num_oversamples)
        ss_steps = {
            **self.sparse_structure_sampler_params,
            **sparse_structure_sampler_params,
        }.get("steps")
        with self.inject_sampler_multi_image(
            "sparse_structure_sampler", len(images), ss_steps, mode=mode
        ):
            coords = self.sample_sparse_structure(
                cond, num_oversamples, sparse_structure_sampler_params
            )
            coords = (
                coords
                if num_oversamples <= num_samples
                else self.select_coords(coords, num_samples)
            )
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get("steps")
        with self.inject_sampler_multi_image(
            "slat_sampler", len(images), slat_steps, mode=mode
        ):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @torch.no_grad()
    def run_multi_image_with_voxel_count(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["gaussian"],
        preprocess_image: bool = True,
        mode: Literal["stochastic", "multidiffusion"] = "multidiffusion",
        num_oversamples: int = 1,
        voxel_threshold: int = 25000,
    ) -> tuple[dict, int]:
        """
        Run the pipeline with multiple images as condition and adjust texture steps based on voxel count.
        
        If occupied voxels > voxel_threshold, use current slat_steps.
        Otherwise, increase slat_steps by 50% for better texture quality.

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
            voxel_threshold (int): Threshold for adjusting texture generation steps.
            
        Returns:
            tuple: (outputs dict, number of occupied voxels)
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond["neg_cond"] = cond["neg_cond"][:1]
        torch.manual_seed(seed)
        num_oversamples = max(num_samples, num_oversamples)
        ss_steps = {
            **self.sparse_structure_sampler_params,
            **sparse_structure_sampler_params,
        }.get("steps")
        with self.inject_sampler_multi_image(
            "sparse_structure_sampler", len(images), ss_steps, mode=mode
        ):
            coords = self.sample_sparse_structure(
                cond, num_oversamples, sparse_structure_sampler_params
            )
            coords = (
                coords
                if num_oversamples <= num_samples
                else self.select_coords(coords, num_samples)
            )
        
        # Count occupied voxels
        num_voxels = len(coords)
        
        # Adjust slat_steps based on voxel count
        base_slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get("steps")
        if num_voxels > voxel_threshold:
            adjusted_slat_steps = base_slat_steps
            print(f"Voxel count {num_voxels} > {voxel_threshold}: Using standard texture steps ({adjusted_slat_steps})")
        else:
            adjusted_slat_steps = int(base_slat_steps * 1.5)
            print(f"Voxel count {num_voxels} <= {voxel_threshold}: Using increased texture steps ({adjusted_slat_steps})")
        
        # Update slat_sampler_params with adjusted steps
        adjusted_slat_sampler_params = {**slat_sampler_params, "steps": adjusted_slat_steps}
        
        with self.inject_sampler_multi_image(
            "slat_sampler", len(images), adjusted_slat_steps, mode=mode
        ):
            slat = self.sample_slat(cond, coords, adjusted_slat_sampler_params)
        
        outputs = self.decode_slat(slat, formats)
        return outputs, num_voxels

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class TrellisVGGTTo3DPipeline(TrellisImageTo3DPipeline):
    def get_ss_cond(self, image_cond: torch.Tensor, aggregated_tokens_list: List, num_samples: int) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image_cond: The image conditioning tensor
            aggregated_tokens_list: List of aggregated tokens from VGGT
            num_samples: Number of samples

        Returns:
            dict: The conditioning information
        """
        cond = self.sparse_structure_vggt_cond(aggregated_tokens_list, image_cond)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def get_slat_cond(self, image_cond: torch.Tensor, aggregated_tokens_list: List, num_samples: int) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image_cond: The image conditioning tensor
            aggregated_tokens_list: List of aggregated tokens from VGGT
            num_samples: Number of samples

        Returns:
            dict: The conditioning information
        """
        b, n, _, _ = aggregated_tokens_list[0].shape
        cond = self.slat_vggt_cond(aggregated_tokens_list, image_cond).reshape(b, n, -1, 1024)
        cond = [c.squeeze(1) for c in cond.split(1, dim=1)]
        neg_cond = [torch.zeros_like(c) for c in cond]
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    
    @torch.no_grad()
    def vggt_feat(self, image: Union[torch.Tensor, list[Image.Image]]) -> List:
        """
        Encode the image using VGGT.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            tuple: (aggregated_tokens_list, image_tensor)
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            image = F.interpolate(image, self.default_image_resolution, mode='bilinear', align_corners=False)
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.default_image_resolution, self.default_image_resolution), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.VGGT_dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                aggregated_tokens_list, _ = self.VGGT_model.aggregator(image[None])
        
        return aggregated_tokens_list, image

    def run(
        self,
        image: Union[torch.Tensor, list[Image.Image]],
        coords: torch.Tensor = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['gaussian'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        torch.manual_seed(seed)
        aggregated_tokens_list, _ = self.vggt_feat(image)
        b, n, _, _ = aggregated_tokens_list[0].shape
        image_cond = self.encode_image(image).reshape(b, n, -1, 1024)
        
        ss_flow_model = self.models['sparse_structure_flow_model']
        ss_cond = self.get_ss_cond(image_cond[:, :, 5:], aggregated_tokens_list, num_samples)
        # Sample structured latent
        ss_sampler_params = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}
        reso = ss_flow_model.resolution
        ss_noise = torch.randn(num_samples, ss_flow_model.in_channels, reso, reso, reso).to(self.device)
        ss_latent = self.sparse_structure_sampler.sample(
            ss_flow_model,
            ss_noise,
            **ss_cond,
            **ss_sampler_params,
            verbose=True
        ).samples

        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(ss_latent)>0)[:, [0, 2, 3, 4]].int()

        slat_cond = self.get_slat_cond(image_cond, aggregated_tokens_list, num_samples)
        # Handle multi-image conditioning by using inject_sampler_multi_image
        # get_slat_cond returns lists when there are multiple images
        num_images = len(image) if isinstance(image, list) else 1
        if num_images > 1:
            slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps', 20)
            with self.inject_sampler_multi_image('slat_sampler', num_images, slat_steps, mode=mode):
                slat = self.sample_slat(slat_cond, coords, slat_sampler_params)
        else:
            slat = self.sample_slat(slat_cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats), coords, ss_noise

    @staticmethod
    def from_pretrained(path: str) -> "TrellisVGGTTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisVGGTTo3DPipeline, TrellisVGGTTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisVGGTTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args
        new_pipeline.VGGT_dtype = torch.float32
        VGGT_model = VGGT.from_pretrained("Stable-X/vggt-object-v0-1")
        new_pipeline.VGGT_model = VGGT_model.to(new_pipeline.device)
        del new_pipeline.VGGT_model.depth_head
        del new_pipeline.VGGT_model.track_head
        new_pipeline.VGGT_model.eval()

        new_pipeline.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet',
            trust_remote_code=True
        ).to(new_pipeline.device)
        new_pipeline.birefnet_model.eval()
        
        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
