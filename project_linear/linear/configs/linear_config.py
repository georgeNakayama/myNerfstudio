
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig, ValidParamGroupsConfig, AdamWOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig, LogDecaySchedulerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from linear.models.linear_mipnerf import LinearMipNerfModelConfig
from linear.models.linear_nerfacto import LinearNerfactoModelConfig
from linear.models.linear_vanilla_nerf import LinearVanillaModelConfig
from nerfstudio.engine.schedulers import VanillaNeRFDecaySchedulerConfig
linear_mip_nerf = MethodSpecification(
    TrainerConfig(
    method_name="linear-mipnerf",
    steps_per_train_image=5000,
    steps_per_eval_image=5000,
    steps_per_test_all_images=100000000000000,
    max_num_iterations=200001,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=BlenderDataParserConfig(), train_num_rays_per_batch=4096),
        model=LinearMipNerfModelConfig(
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=1024,
            near_plane=2.0,
            far_plane=6.0,
            color_mode="midpoint"
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=5e-4, weight_decay=1e-5),
            "scheduler": LogDecaySchedulerConfig(lr_final=1e-6, lr_delay_steps=2500, lr_delay_mult=0.01, max_steps=1000000),
        }
    }),
    description="linear mip model"
)

linear_nerfacto = MethodSpecification(
    TrainerConfig(
    method_name="linear-nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    steps_per_train_image=500,
    steps_per_eval_image=500,
    steps_per_test_all_images=100000,
    steps_per_eval_all_images=5000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(), 
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="off",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
            ),
        ),
        model=LinearNerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            disable_scene_contraction=True,
            proposal_initial_sampler="uniform",
            near_plane=2.0,
            far_plane=6.0,
            color_mode="midpoint",
            background_color="white"
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",),
    description="linear nerfacto model"
)


blender_nerfacto = MethodSpecification(
    TrainerConfig(
    method_name="blender_nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="off",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15, near_plane=2, far_plane=6),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
    ),
    description="blender nerfacto model"
)

linear_vanilla_nerf = MethodSpecification(
    TrainerConfig(
        method_name="linear-vanilla-nerf",
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
            ),
            model=LinearVanillaModelConfig(),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4),
                "scheduler": VanillaNeRFDecaySchedulerConfig(),
            },
            "temporal_distortion": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
    ),
    description="linear vanilla NeRF"
)