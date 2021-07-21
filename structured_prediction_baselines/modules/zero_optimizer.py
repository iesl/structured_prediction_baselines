import torch
from typing import Any, Dict, List, Tuple, Union, Optional
from allennlp.training.optimizers import Optimizer, SgdOptimizer

# import copy
# import logging
# import re
# import math
# from typing import Any, Dict, List, Tuple, Union, Optional

# from overrides import overrides
# import torch
# import transformers

# from allennlp.common import Params, Registrable, Lazy
# from allennlp.common.checks import ConfigurationError
# logger = logging.getLogger(__name__)

@Optimizer.register("zero")
class ZeroOptimizer(SgdOptimizer):
    """
    Registered as an `Optimizer` with name "zero" 
    as it's purpose is to "not" update the model.
    """
    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        lr: float,
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        momentum: float = 0.0,
        dampening: float = 0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(
            model_parameters=model_parameters,
            lr=lr,
            parameter_groups=parameter_groups,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    def step(self):
        """
        Doesn't to any operation.
        """
        pass
   
