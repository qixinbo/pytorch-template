import logging

from accelerate import notebook_launcher as accelerate_notebook_launcher
from accelerate.utils import set_seed

from pytorch_accelerated.trainer import Trainer, TrainerPlaceholderValues
from . import _version

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

logger.info("----------- Attention-------------")
logger.info("Since the APIs of pytorch_accelerated are not stable currently, we are using the in-built one")
logger.info("----------------------------------")
logger.info("Setting random seeds")
set_seed(42)
notebook_launcher = accelerate_notebook_launcher

__version__ = _version.get_versions()["version"]
