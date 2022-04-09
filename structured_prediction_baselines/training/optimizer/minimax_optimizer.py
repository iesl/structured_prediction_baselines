import logging
from collections import defaultdict
from collections.abc import MutableMapping
from typing import (
    Optional,
    List,
    Dict,
    Tuple,
    Any,
    Callable,
    Iterator,
    cast,
    Literal, Union,
)

import torch
from allennlp.common import Lazy, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.training.optimizers import (
    Optimizer,
    ParameterGroupsType,
)

from structured_prediction_baselines.common import ModelMode, OptimizerMode

logger = logging.getLogger(__name__)

MODE_LITERALS_TYPE = Literal[
    ModelMode.UPDATE_TASK_NN.value, ModelMode.UPDATE_SCORE_NN.value
]

OPTIMIZER_LITERALS_TYPE = Literal[
    OptimizerMode.FEATURE_NET.value, OptimizerMode.NON_FEATURE_NET.value, OptimizerMode.FULL.value
]


class MiniMaxOptimizer(Optimizer, MutableMapping, Registrable):
    """
    Holds multiple optimizers as dictionary with string keys.
    Each it behaves as `torch.optim.Optimizer` but all the methods take in
    an extra parameter `opt_key` (aka `model_mode`) to identify the optimizer to act upon.


    Although we inherit from `MultiOptimizer` we do not assign parameters like `MultiOptimizer`.
    `MultiOptimizer` assigns parameters to individual optimizers by looking for `optimizer_name` key
    in the parameter group values. We will instead assign parameters by querying the model for appropriate parameters.
    """

    default_implementation = "minimax"


@MiniMaxOptimizer.register("minimax")
class MiniMaxDefaultOptimizer(MiniMaxOptimizer, MutableMapping):

    def __init__(
            self,
            model_parameters: List[Tuple[str, torch.nn.Parameter]],
            optimizers: Dict[
                MODE_LITERALS_TYPE,
                Lazy[Optimizer],
            ],
            parameter_groups: Dict[
                MODE_LITERALS_TYPE,
                ParameterGroupsType
            ] = None,
    ):
        # split the parameters and assign them to the correct optimizer
        # Note: If a parameter does not have the model_mode attribute set,
        # then that parameter will not be assigned to any optimizer

        if parameter_groups is None:
            parameter_groups = {}

        unassigned_params = []
        named_params_: Dict[
            MODE_LITERALS_TYPE,
            List[Tuple[str, torch.nn.Parameter]],
        ] = defaultdict(list)

        for n, p in model_parameters:
            if not ModelMode.hasattr_model_mode(p):
                unassigned_params.append(n)

                continue
            mode_name = ModelMode.getattr_model_mode(p).value

            if mode_name not in optimizers:
                unassigned_params.append(n)

                continue
            mode_name_: MODE_LITERALS_TYPE = cast(
                MODE_LITERALS_TYPE, mode_name
            )  # no runtime effect
            named_params_[mode_name_].append((n, p))

        logger.info("Optimizer assignements are as follows.")

        for mode_name, params in named_params_.items():
            logger.info(
                f"Following parameters have been assigned to the {mode_name} optimizer"
            )

            for n, p in params:
                logger.info(f"{n}")
        logger.info(
            "Following parameters have not been assigned to any optimizer, hence will not be updated"
        )

        for n in unassigned_params:
            logger.info(f"{n}")

        self.optimizers = {
            mode_name: lazy_optimizer.construct(
                model_parameters=named_params_[mode_name],
                parameter_groups=parameter_groups.get(mode_name)
            )
            for mode_name, lazy_optimizer in optimizers.items()
        }

        super().__init__([v for k, v in model_parameters], {})

    def __getitem__(self, key: MODE_LITERALS_TYPE) -> torch.optim.Optimizer:
        return self.optimizers[key]

    def __setitem__(
            self, key: MODE_LITERALS_TYPE, value: torch.optim.Optimizer
    ) -> None:
        self.optimizers[key] = value

    def __delitem__(self, key: MODE_LITERALS_TYPE) -> None:
        del self.optimizers[key]

    def __iter__(self) -> Iterator[torch.optim.Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def zero_grad(
            self, model_mode: Optional[str] = None, set_to_none: bool = False
    ) -> None:
        if model_mode is not None:
            self.optimizers[model_mode].zero_grad(set_to_none=set_to_none)
        else:
            for k, v in self.optimizers.items():
                v.zero_grad(set_to_none=set_to_none)

    def step(
            self,
            model_mode: Optional[str] = None,
            closure: Optional[Dict[str, Callable]] = None,
    ) -> None:
        if model_mode is not None:
            self.optimizers[model_mode].step(closure=closure)
        else:
            for k, v in self.optimizers.items():
                v.step(closure=closure[k] if closure is not None else None)

    def state_dict(self) -> Dict:
        """
        Creates an object `optimizer_state_dict`, which is a dictionary mapping an optimizer key to its
        `state_dict`. This dictionary is used as the value for 'optimizer' in the 'training_states' dictionary in
        the `gradient_descent` `Trainer`, e.g.
        ```
        "optimizer" : {
            "optimizer1": `optimizer1_state_dict`,
            "optimizer2": `optimizer2_state_dict`
        }.
        ```
        """
        optimizer_state_dict = {
            f"{optimizer_key}_optimizer": optimizer.state_dict()
            for optimizer_key, optimizer in self.optimizers.items()
        }

        return optimizer_state_dict

    def load_state_dict(self, training_state: Dict[str, Any]) -> None:
        """
        Loads each optimizer's `state_dict`.
        """

        for optimizer_key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(
                training_state[f"{optimizer_key}_optimizer"]
            )


@MiniMaxOptimizer.register("minimax_multimodal")
class MiniMaxMultimodalOptimizer(MiniMaxOptimizer, MutableMapping):
    """
    Holds multiple optimizers for each mode as dictionary with string keys.
    Each it behaves as `torch.optim.Optimizer` but all the methods take in
    two extra parameters `model_mode` and `opt_key` to identify the optimizer to act upon.


    Although we inherit from `MultiOptimizer` we do not assign parameters like `MultiOptimizer`.
    `MultiOptimizer` assigns parameters to individual optimizers by looking for `optimizer_name` key
    in the parameter group values. We will instead assign parameters by querying the model for appropriate parameters.

    Note:
        We are skipping the complex parameter grouping logic for now. It can be implemented if needed.
    """

    def __init__(
            self,
            model_parameters: List[Tuple[str, torch.nn.Parameter]],
            optimizers: Dict[
                MODE_LITERALS_TYPE,
                Lazy[Optimizer],
            ],
            shared_feature_net: bool = False,
            parameter_groups: ParameterGroupsType = None,
    ):
        # split the parameters and assign them to the correct optimizer
        # Note: If a parameter does not have the model_mode attribute set,
        # then that parameter will not be assigned to any optimizer

        if parameter_groups is not None:
            raise ConfigurationError("parameter_groups are not supported.")

        unassigned_params = []
        named_params_: Dict[
            MODE_LITERALS_TYPE,
            Dict[
                OPTIMIZER_LITERALS_TYPE,
                List[Tuple[str, torch.nn.Parameter]]
            ],
        ] = defaultdict(lambda: defaultdict(list))

        for n, p in model_parameters:
            if not ModelMode.hasattr_model_mode(p):
                unassigned_params.append(n)
                continue
            mode_name = ModelMode.getattr_model_mode(p).value
            optimizer_mode = OptimizerMode.getattr_optimizer_mode(p).value
            optimizer_key = self.join_optimizer_key(mode_name, optimizer_mode)
            if optimizer_key not in optimizers and mode_name not in optimizers:
                unassigned_params.append(n)
                continue

            mode_name_: MODE_LITERALS_TYPE = cast(
                MODE_LITERALS_TYPE, mode_name
            )  # no runtime effect
            optimizer_mode_: OPTIMIZER_LITERALS_TYPE = cast(
                OPTIMIZER_LITERALS_TYPE, optimizer_mode
            )
            named_params_[mode_name_][optimizer_mode_].append((n, p))

        logger.info("Optimizer assignements are as follows.")

        for mode_name, optimizer_params in named_params_.items():
            for optimizer_mode, params in optimizer_params.items():
                logger.info(
                    f"Following parameters have been assigned to the {mode_name}({optimizer_mode}) optimizer"
                )

                for n, p in params:
                    logger.info(f"{n}")

        logger.info(
            "Following parameters have not been assigned to any optimizer, hence will not be updated"
        )

        for n in unassigned_params:
            logger.info(f"{n}")

        self.optimizers: Dict[
            MODE_LITERALS_TYPE,
            Dict[OPTIMIZER_LITERALS_TYPE, Optimizer]
        ] = defaultdict(dict)

        for key, lazy_optimizer in optimizers.items():
            mode_name, optimizer_mode = self.split_optimizer_key(key)
            if optimizer_mode == OptimizerMode.FULL.value:
                optimizer_parameters_ = sum(named_params_[mode_name].values(), [])
            else:
                optimizer_parameters_ = named_params_[mode_name][optimizer_mode]
            self.optimizers[mode_name][optimizer_mode] = lazy_optimizer.construct(
                    model_parameters=optimizer_parameters_
                )

        if shared_feature_net:
            self.optimizers[ModelMode.UPDATE_SCORE_NN.value][OptimizerMode.FEATURE_NET.value] = self.optimizers[ModelMode.UPDATE_TASK_NN.value][OptimizerMode.FEATURE_NET.value]

        super().__init__([v for k, v in model_parameters], {})

    def __getitem__(self, key: MODE_LITERALS_TYPE,
                    opt_key: OPTIMIZER_LITERALS_TYPE = None) -> Union[Dict[Any, Optimizer], Optimizer]:
        if opt_key:
            return self.optimizers[key][opt_key]
        return self.optimizers[key]

    def __setitem__(
            self, key: MODE_LITERALS_TYPE, value: torch.optim.Optimizer,
            opt_key: OPTIMIZER_LITERALS_TYPE = OptimizerMode.FULL.value
    ) -> None:
        self.optimizers[key][opt_key] = value

    def __delitem__(self, key: MODE_LITERALS_TYPE, opt_key: OPTIMIZER_LITERALS_TYPE = OptimizerMode.FULL.value) -> None:
        del self.optimizers[key][opt_key]

    def __iter__(self) -> Iterator[torch.optim.Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def zero_grad(
            self, model_mode: Optional[str] = None, set_to_none: bool = False
    ) -> None:
        if model_mode is not None:
            for k, v in self.optimizers[model_mode].items():
                v.zero_grad(set_to_none=set_to_none)
        else:
            for k, v in self.optimizers.items():
                for o in v.values():
                    o.zero_grad(set_to_none=set_to_none)

    def step(
            self,
            model_mode: Optional[str] = None,
            closure: Optional[Dict[str, Callable]] = None,
    ) -> None:
        if model_mode is not None:
            for k, v in self.optimizers[model_mode].items():
                v.step(closure=closure)
        else:
            for k, v in self.optimizers.items():
                for o in v.values():
                    o.step(closure=closure[k] if closure is not None else None)

    def state_dict(self) -> Dict:
        """
        Creates an object `optimizer_state_dict`, which is a dictionary mapping an optimizer key to its
        `state_dict`. This dictionary is used as the value for 'optimizer' in the 'training_states' dictionary in
        the `gradient_descent` `Trainer`, e.g.
        ```
        "optimizer" : {
            "optimizer1": `optimizer1_state_dict`,
            "optimizer2": `optimizer2_state_dict`
        }.
        ```
        """
        optimizer_state_dict = {}
        for model_mode, mode_optimizers in self.optimizers.items():
            for optimizer_key, optimizer in mode_optimizers.items():
                optimizer_state_dict[f"{model_mode}_{optimizer_key}_optimizer"] = optimizer.state_dict()

        return optimizer_state_dict

    def load_state_dict(self, training_state: Dict[str, Any]) -> None:
        """
        Loads each optimizers' `state_dict`.
        """

        for model_mode, mode_optimizers in self.optimizers.items():
            for optimizer_key, optimizer in mode_optimizers.items():
                optimizer.load_state_dict(
                    training_state[f"{model_mode}_{optimizer_key}_optimizer"]
                )

    @staticmethod
    def join_optimizer_key(mode_name: str, optimizer_mode: str):
        if optimizer_mode == "full":
            return mode_name

        return '_'.join([mode_name, optimizer_mode])

    @staticmethod
    def split_optimizer_key(optimizer_key: str):
        split_keys = optimizer_key.split("_")
        if len(split_keys) == 2:
            return optimizer_key, OptimizerMode.FULL.value

        mode_name = '_'.join(split_keys[:2])
        optimizer_mode = '_'.join(split_keys[2:])
        return mode_name, optimizer_mode
