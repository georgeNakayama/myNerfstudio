
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from myproject.cimle_trainer import cIMLETrainerConfig
from myproject.vanilla_cimle_nerf import cIMLEVanillaModelConfig
from myproject.cimle_pipeline import cIMLEPipelineConfig
from myproject.cimle_datamanager import cIMLEDataManagerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from myproject.sparse_scannet_dataparser import SparseScannetDataParserConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
CIMLE_CH=32

cimle_config = MethodSpecification(
    config=cIMLETrainerConfig(
        method_name="cimle-vanilla-nerf",
        pipeline=cIMLEPipelineConfig(
            datamanager=cIMLEDataManagerConfig(
                dataparser=SparseScannetDataParserConfig(train_json_name="transforms_train_wo_663_932.json", val_json_name="transforms_test.json"),
            ),
            model=cIMLEVanillaModelConfig(cimle_ch=CIMLE_CH, head_only=True),
            cimle_sample_num=20,
            cimle_cache_interval=2000,
            cimle_ch=CIMLE_CH,
            cimle_num_rays_to_test=100*100
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
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

sparse_scannet = DataParserSpecification(config=SparseScannetDataParserConfig())