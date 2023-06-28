
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig, ValidParamGroupsConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from myproject.engines.trainers.cimle_trainer import cIMLETrainerConfig
from myproject.models.vanilla_cimle_nerf import cIMLEVanillaModelConfig
from myproject.models.cimle_nerfacto import cIMLENerfactoModelConfig
from myproject.pipelines.cimle_pipeline import cIMLEPipelineConfig
from myproject.data.datamanagers.cimle_datamanager import cIMLEDataManagerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from myproject.data.dataparsers.sparse_scannet_dataparser import SparseScannetDataParserConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from myproject.data.dataparsers.llff_dataparser import LLFFDataParserConfig

CIMLE_CH=32

cimle_vanilla_nerf = MethodSpecification(
    config=cIMLETrainerConfig(
        method_name="cimle-vanilla-nerf",
        pipeline=cIMLEPipelineConfig(
            datamanager=cIMLEDataManagerConfig(
                dataparser=SparseScannetDataParserConfig(train_json_name="transforms_train_wo_663_932.json", val_json_name="transforms_test.json"),
            ),
            model=cIMLEVanillaModelConfig(cimle_ch=CIMLE_CH, head_only=True),
            cimle_sample_num=20,
            cimle_cache_interval=100000,
            cimle_ch=CIMLE_CH,
            cimle_num_rays_to_test=200*200
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
            "cimle": {
                "optimizer": RAdamOptimizerConfig(lr=5e-5, eps=1e-08),
                "scheduler": None,
            },
            "temporal_distortion": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
    ),
    description="cimle_vanilla_nerf configuration file"
)

# CIMLE_CH=32
# cimle_nerfacto = MethodSpecification(
#     config=cIMLETrainerConfig(
#         method_name="cimle-nerfacto",
#         pipeline=cIMLEPipelineConfig(
#             datamanager=cIMLEDataManagerConfig(
#                 dataparser=SparseScannetDataParserConfig(train_json_name="transforms_train.json", val_json_name="transforms_val.json"),
#             ),
#             model=cIMLENerfactoModelConfig(cimle_ch=CIMLE_CH, color_cimle=False),
#             cimle_sample_num=20,
#             cimle_cache_interval=1000000,
#             cimle_ch=CIMLE_CH,
#             cimle_num_rays_to_test=200*200,
#         ),
#         optimizers={
#             "proposal_networks": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
#             },
#             "fields": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
#             },
#             "cimle": {
#                 "optimizer": RAdamOptimizerConfig(lr=5e-5, eps=1e-08),
#                 "scheduler": None,
#             },
#         },
#     ),
#     description="cimle_vanilla_nerf configuration file"
# )


CIMLE_CH=32
cimle_nerfacto = MethodSpecification(
    TrainerConfig(
    method_name="cimle-nerfacto",
    steps_per_eval_batch=5000,
    steps_per_eval_image=5000,
    steps_per_train_image=1000,
    steps_per_test_all_images=10000,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
            ),
        ),
        model=cIMLENerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            cimle_ch=CIMLE_CH, 
            color_cimle=False,
            cimle_sample_num=20,
            cimle_cache_interval=5000,
            cimle_num_rays_to_test=-1,
            cimle_injection_type="add",
            cimle_activation="relu",
            cimle_pretrain=False
                                  ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "proposal_networks.cimle": {
            "optimizer": AdamOptimizerConfig(lr=5e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields.cimle": {
            "optimizer": AdamOptimizerConfig(lr=5e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=200000),
        },
        "valid_param_groups": ValidParamGroupsConfig(valid_pgs=["proposal_networks","fields", "camera_opt", "fields.cimle", "proposal_networks.cimle"])
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="wandb",
    ),
    description="cimle nerfacto model"
)

cimle_nerfacto_pretrain = MethodSpecification(
    TrainerConfig(
    method_name="cimle-nerfacto-pretrain",
    steps_per_eval_batch=5000,
    steps_per_eval_image=5000,
    steps_per_train_image=1000,
    steps_per_test_all_images=10000,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=LLFFDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
            ),
        ),
        model=cIMLENerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            cimle_ch=CIMLE_CH, 
            color_cimle=False,
            cimle_sample_num=20,
            cimle_cache_interval=5000,
            cimle_num_rays_to_test=-1,
            cimle_injection_type="add",
            cimle_activation="relu",
            cimle_pretrain=True
                                  ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "proposal_networks.cimle": {
            "optimizer": AdamOptimizerConfig(lr=5e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields.cimle": {
            "optimizer": AdamOptimizerConfig(lr=5e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=200000),
        },
        "valid_param_groups": ValidParamGroupsConfig(valid_pgs=["proposal_networks", "fields", "camera_opt"])
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="wandb",
    ),
    description="cimle nerfacto model pretraining"
)
