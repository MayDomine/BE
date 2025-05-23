#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import os
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import (
    IS_REPLICA_EXPERT_DATA_PARALLEL,
    IS_REPLICA_ZERO_PARALLEL,
    IS_TENSOR_EXPERT_DATA_PARALLEL,
    IS_TENSOR_ZERO_PARALLEL,
    IS_WEIGHT_EXPERT_DATA_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
    ParallelMode,
)
from internlm.core.context import global_context as gpc
from internlm.core.context.random import set_mode
from internlm.core.naive_amp import (
    NaiveAMPModel,
    set_fp32_attr_to_module,
    unwrap_naive_amp,
)
from internlm.core.parallel.comm.isp import (
    EmbeddingWeightParallelCommunicator,
    HeadWeightParallelCommunicator,
    ISPCommModelConfig,
    ISPCommunicator,
    ISPCommunicatorSchedulerHook,
    ISPCommunicatorWrapper,
)
from internlm.core.parallel.comm.tensor import (
    EmbeddingSequenceParallelCommunicator,
    EmbeddingTensorParallelCommunicator,
    HeadSequenceParallelCommunicator,
    HeadTensorParallelCommunicator,
    LinearRole,
    MoESequenceParallelCommunicator,
    SequenceParallelCommunicator,
    TensorParallelCommunicator,
)
from internlm.core.parallel.comm.zero import ParamAsyncBcastHandler
from internlm.core.trainer import TrainState
from internlm.data.utils import unpack_type_ids
from internlm.model.builder import create_model
from internlm.model.metrics import SchedulerMetricHook
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.linear import (
    ColumnParallelLinear,
    GroupedColumnLinear,
    GroupedRowLinear,
    GroupedWPLinear,
    ParallelLinearWithCommExt,
    RewardModelLinear,
    RowParallelLinear,
    ScaleColumnParallelLinear,
    new_linear,
)
from internlm.model.modules.norm import new_layer_norm
from internlm.model.moe import Experts, MoE
from internlm.model.moe.moe import Qwen2MoE
from internlm.model.ops.norm import RMSNorm
from internlm.model.registry import register_model_initializer
from internlm.monitor import set_env_var
from internlm.monitor.monitor import monitor_manager as mm
from internlm.solver.optimizer import (
    FSDPadaptOptimizer,
    HybridZeroOptimizer,
    HybridZeroOptimizer_v2,
)
from internlm.solver.optimizer.compatible_adamw import new_compatible_adamw
from internlm.solver.schedulers.beta2_scheduler import Beta2Scheduler
from internlm.solver.schedulers.lr_scheduler import FineTuneCosineAnnealingWarmupLR
from internlm.train.utils import create_param_groups, map_param_block, timeout_input
from internlm.utils.common import DummyProfile, SchedulerHook, get_current_device
from internlm.utils.logger import get_logger
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.parallel import (
    is_replica_expert_data_parallel_parameter,
    is_replica_zero_parallel_parameter,
    is_tensor_expert_data_parallel_parameter,
    is_tensor_zero_parallel_parameter,
    is_using_isp,
    is_weight_expert_data_parallel_parameter,
    is_weight_zero_parallel_parameter,
    sync_model_param,
    sync_model_replica_param_group,
)
from internlm.utils.timeout import llm_timeout
from internlm.utils.utils import TensorParallelMode

try:
    import torch_npu
except (ImportError, ModuleNotFoundError):
    pass

IS_INJECTED = "is_injected"

LINEAR2NEWLINEAR_NAME_MAPPING = dict(
    q_proj="wq",
    k_proj="wk",
    v_proj="wv",
    o_proj="wo",
    gate_proj="w1",
    down_proj="w2",
    up_proj="w3",
    lm_head="head",
    W_pack="wqkv",
)

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()


def set_param_unique_tracking_name(model):
    for chunk_id, chunk in enumerate(unwrap_naive_amp(model)):
        # Important: only works for llama-class models
        childrens = chunk.named_children()
        for _, children in childrens:
            if isinstance(children, nn.ModuleList):
                for idx, block in enumerate(children):
                    for name, child in block.named_modules():
                        if isinstance(child, (ParallelLinearWithCommExt)):
                            full_name = f"{chunk_id}.{idx}.{name}"
                            setattr(
                                child.weight,
                                "tracking_name",
                                f"{full_name}.weight",
                            )
                            if child.bias is not None:
                                setattr(
                                    child.bias,
                                    "tracking_name",
                                    f"{full_name}.bias",
                                )
            else:
                if isinstance(children, Embedding1D):
                    setattr(
                        children.weight,
                        "tracking_name",
                        f"{chunk_id}_embedding.weight",
                    )
                else:
                    setattr(
                        children.weight,
                        "tracking_name",
                        f"{chunk_id}_head.weight",
                    )


def set_fp32_attr_for_model(model: Union[nn.Module, nn.ModuleList]):
    if not isinstance(model, nn.ModuleList):
        model = [model]

    for _chunk in model:
        for _, module in _chunk.named_modules():
            if isinstance(module, (RMSNorm, nn.LayerNorm)) and gpc.config.get("use_fp32_norm", False):
                set_fp32_attr_to_module(module)


def set_parallel_attr_for_param_groups(model: Union[nn.Module, nn.ModuleList]):
    def _check_module_pure_dp_wdp(name, module):  # pylint: disable=W0613
        for param in module.parameters():
            setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

    def _check_module(name, module):
        # layer_norm
        if isinstance(module, (RMSNorm, nn.LayerNorm)):
            for param in module.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

        if isinstance(module, (MoE, Qwen2MoE)):
            for param in module.moe_layer.gate.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)
            if hasattr(module, "coefficient"):
                for param in module.coefficient.parameters():
                    setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

        # embedding and head
        if isinstance(module, (Embedding1D, ScaleColumnParallelLinear)):
            for param in module.parameters():
                if gpc.is_initialized(ParallelMode.WEIGHT) and is_using_isp():
                    setattr(param, IS_WEIGHT_ZERO_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.TENSOR) and not is_using_isp():
                    setattr(param, IS_TENSOR_ZERO_PARALLEL, True)

        # for moe linear module
        if isinstance(module, nn.Linear) and not isinstance(module, ParallelLinearWithCommExt):
            for param in module.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

        if isinstance(module, Experts):
            for param in module.parameters():
                if (
                    gpc.is_initialized(ParallelMode.TENSOR)
                    and not is_using_isp()
                    and getattr(gpc.config.parallel.expert, "no_tp", False)
                ):
                    setattr(param, IS_REPLICA_EXPERT_DATA_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.TENSOR) and not is_using_isp():
                    setattr(param, IS_TENSOR_EXPERT_DATA_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.WEIGHT) and is_using_isp():
                    setattr(param, IS_WEIGHT_EXPERT_DATA_PARALLEL, True)
        # for non-moe linear module
        elif isinstance(module, ParallelLinearWithCommExt):
            for param in module.parameters():
                if gpc.is_initialized(ParallelMode.TENSOR) and not is_using_isp():
                    setattr(param, IS_TENSOR_ZERO_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.WEIGHT) and is_using_isp():
                    setattr(param, IS_WEIGHT_ZERO_PARALLEL, True)

        # for vit and vit project
        if "vision_tower" in name.lower() or "vision_proj" in name.lower():
            for param in module.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

    for _chunk in unwrap_naive_amp(model):
        # special case for pure dp or pure wdp mode
        if gpc.get_world_size(ParallelMode.DATA) == gpc.get_world_size(ParallelMode.GLOBAL) and gpc.get_world_size(
            ParallelMode.WEIGHT_DATA
        ) == gpc.get_world_size(ParallelMode.GLOBAL):
            _check_module_func = _check_module_pure_dp_wdp
        else:
            _check_module_func = _check_module
        # set param parallel attribute
        for name, module in _chunk.named_modules():
            _check_module_func(name, module)

        for name, param in _chunk.named_parameters():
            assert (
                is_replica_zero_parallel_parameter(param)
                or is_tensor_zero_parallel_parameter(param)
                or is_weight_zero_parallel_parameter(param)
                or is_tensor_expert_data_parallel_parameter(param)
                or is_weight_expert_data_parallel_parameter(param)
                or is_replica_expert_data_parallel_parameter(param)
            ), f"parameter with name: {name} has no parallel attribution."


@llm_timeout(func_name="initialize_model")
def initialize_model(pre_process_func: Optional[Callable] = None, post_process_func: Optional[Callable] = None):
    """
    Initialize model with Automatic Mixed Precision.
    Returns:
        torch.nn.Module:
            The neural network model to be trained or evaluated.
    """
    if pre_process_func:
        pre_process_output = pre_process_func()

    register_model_initializer()

    model = create_model(model_type=gpc.config.model_type)

    if post_process_func:
        post_process_func(pre_process_output)

    return inject_model(model)


def inject_model(model):
    """
    Inject model with Automatic Mixed Precision.

    Args:
        torch.nn.Module:
            The bare neural network model to be trained or evaluated.

    Returns:
        torch.nn.Module:
            The injected neural network model to be trained or evaluated.
    """
    if hasattr(model, IS_INJECTED) and getattr(model, IS_INJECTED):
        return model

    inject_model_helper(model, inject_info=gpc.config.model.get("inject_info", None))

    # should be set before NaiveAMPModel
    set_fp32_attr_for_model(model)

    if isinstance(model, nn.ModuleList):
        model = nn.ModuleList(
            [
                NaiveAMPModel(
                    model=_m,
                    output_to_fp32=False,  # manually controlled by interleaved pipleline scheduler
                    dtype=gpc.config.model.get("dtype", torch.half),
                    sync_buffer=False,
                )
                for _m in model
            ]
        )
    else:
        model = NaiveAMPModel(
            model=model,
            output_to_fp32=False,
            dtype=gpc.config.model.get("dtype", torch.half),
            sync_buffer=False,
        )

    set_parallel_attr_for_param_groups(model)

    # This sync is very important, cause the model weights kept in optimizer are copied
    # from the origin parameters in the memory, so we should make sure the dp sync
    # does not influence the model weights in optimizer be different with the origin parameters.
    sync_model_param(model)

    # This function is needed to make sure parameters that are not splitted by tensor parallelism are
    # the same across tensor parallelism.
    sync_model_replica_param_group(model)

    # Change random state mode to ParallelMode.DATA after model is built, guaranteeing the random
    # state in the same dp group are all the same.
    random_mode = ParallelMode.WEIGHT_DATA if is_using_isp() else ParallelMode.DATA
    set_mode(random_mode)

    # set is_injected flag
    setattr(model, "IS_INJECTED", True)

    return model


_T = TypeVar("_T")


def _submodule_filter(model: Union[nn.Module, nn.ModuleList], target_cls: Union[_T, Tuple[_T]]) -> Iterable[_T]:
    for _chunk in unwrap_naive_amp(model):
        for _module in _chunk.modules():
            if not isinstance(_module, target_cls):
                continue

            yield _module


def initialize_parallel_communicator(model: Union[nn.Module, nn.ModuleList]):
    """
    Initialize communicator for isp tensor parallel mode.

    Args:
        model (:class:`torch.nn.Module`): Your model instance to be trained or evaluated.

    Returns:
        An isp communicator for managing comp/comm overlap.
    """
    isp_communicator_wrapper = None
    _retain_out_sharded = gpc.config.model.get("parallel_output", True)

    if is_using_isp():
        isp_communicator = ISPCommunicator(
            model,
            ISPCommModelConfig(
                gpc.config.model.dtype,
                get_current_device(),
                gpc.config.model.checkpoint,
            ),
            gpc.config.parallel.weight.overlap,
            gpc.get_group(ParallelMode.WEIGHT),
            is_moe=False,
        )
        # register communicator for isp column parallel linear.
        ColumnParallelLinear.register_cls_communicator(isp_communicator)
        # row parallel linear will not be used.
        RowParallelLinear.register_cls_communicator(None)
        _head_communicator = HeadWeightParallelCommunicator(
            weight_process_group=gpc.get_group(ParallelMode.WEIGHT),
            seq_process_group=gpc.get_group(ParallelMode.TENSOR),
            retain_out_sharded=_retain_out_sharded,
        )
        _embedding_communicator = EmbeddingWeightParallelCommunicator(ParallelMode.WEIGHT)

        if gpc.config.model.get("num_experts", 1) > 1:
            # register communicator for moe isp column parallel linear.
            # NOTE: this wil overwrite registed communicator
            moe_isp_communicator = ISPCommunicator(
                model,
                ISPCommModelConfig(
                    gpc.config.model.dtype,
                    get_current_device(),
                    gpc.config.model.checkpoint,
                ),
                gpc.config.parallel.expert_weight.overlap,
                gpc.get_group(ParallelMode.EXPERT_WEIGHT),
                is_moe=True,
            )
            for moe in _submodule_filter(model, Experts):
                for column_linear in _submodule_filter(moe, (ColumnParallelLinear, GroupedWPLinear)):
                    column_linear.register_communicator(moe_isp_communicator)
                for row_linear in _submodule_filter(moe, RowParallelLinear):
                    row_linear.register_communicator(None)

            isp_communicator_wrapper = ISPCommunicatorWrapper([isp_communicator, moe_isp_communicator])
        else:
            isp_communicator_wrapper = ISPCommunicatorWrapper([isp_communicator])

    # register communictor for mtp/msp/fsp linear.

    # tensor parallel
    if gpc.config.parallel.tensor.mode == TensorParallelMode.mtp.name:
        ColumnParallelLinear.register_cls_communicator(
            TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.COLUMN)
        )
        RowParallelLinear.register_cls_communicator(
            TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.ROW)
        )

        if gpc.config.model.get("num_experts", 1) > 1:
            GroupedColumnLinear.register_cls_communicator(
                TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.COLUMN)
            )
            GroupedRowLinear.register_cls_communicator(
                TensorParallelCommunicator(process_group=gpc.get_group(ParallelMode.TENSOR), role=LinearRole.ROW)
            )
            GroupedWPLinear.register_cls_communicator(None)
            # treat as sequence paralle if no_tp
            if gpc.config.parallel.expert.no_tp:
                _column_communicator = TensorParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.EXPERT_TENSOR), role=LinearRole.COLUMN
                )
                _row_communicator = TensorParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.EXPERT_TENSOR), role=LinearRole.ROW
                )
                for moe in _submodule_filter(model, MoE):
                    # 1. the linear in MoE degrades as no tp communication pattern
                    for column_linear in _submodule_filter(moe, ColumnParallelLinear):
                        column_linear.register_communicator(_column_communicator)
                    for row_linear in _submodule_filter(moe, RowParallelLinear):
                        row_linear.register_communicator(_row_communicator)
                    # 2. register MoESequenceParallelCommunicator for MoE layer
                    MoESequenceParallelCommunicator(ParallelMode.TENSOR, reverse=True).register_module_hook(moe)

        _head_communicator = HeadTensorParallelCommunicator(ParallelMode.TENSOR, _retain_out_sharded)
        _embedding_communicator = EmbeddingTensorParallelCommunicator(ParallelMode.TENSOR)
    # sequence parallel
    if gpc.config.parallel.tensor.mode in (TensorParallelMode.msp.name, TensorParallelMode.fsp.name):
        save_total_input_as_activation = gpc.config.parallel.tensor.mode == TensorParallelMode.msp.name

        ColumnParallelLinear.register_cls_communicator(
            SequenceParallelCommunicator(
                process_group=gpc.get_group(ParallelMode.TENSOR),
                role=LinearRole.COLUMN,
                save_total_input_as_activation=save_total_input_as_activation,
            )
        )
        RowParallelLinear.register_cls_communicator(
            SequenceParallelCommunicator(
                gpc.get_group(ParallelMode.TENSOR),
                role=LinearRole.ROW,
                save_total_input_as_activation=save_total_input_as_activation,
            )
        )
        if gpc.config.model.get("num_experts", 1) > 1:
            GroupedColumnLinear.register_cls_communicator(
                SequenceParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.TENSOR),
                    role=LinearRole.COLUMN,
                    save_total_input_as_activation=save_total_input_as_activation,
                )
            )
            GroupedRowLinear.register_cls_communicator(
                SequenceParallelCommunicator(
                    gpc.get_group(ParallelMode.TENSOR),
                    role=LinearRole.ROW,
                    save_total_input_as_activation=save_total_input_as_activation,
                )
            )
            GroupedWPLinear.register_cls_communicator(None)
            if gpc.config.parallel.expert.no_tp:
                _column_communicator = TensorParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.EXPERT_TENSOR), role=LinearRole.COLUMN
                )
                _row_communicator = TensorParallelCommunicator(
                    process_group=gpc.get_group(ParallelMode.EXPERT_TENSOR), role=LinearRole.ROW
                )
                for moe in _submodule_filter(model, MoE):
                    # 1. the linear in MoE degrades as no tp communication pattern
                    for column_linear in _submodule_filter(moe, ColumnParallelLinear):
                        column_linear.register_communicator(_column_communicator)
                    for row_linear in _submodule_filter(moe, RowParallelLinear):
                        row_linear.register_communicator(_row_communicator)

        _head_communicator = HeadSequenceParallelCommunicator(
            ParallelMode.TENSOR, _retain_out_sharded, save_total_input_as_activation
        )

        _embedding_communicator = EmbeddingSequenceParallelCommunicator(ParallelMode.TENSOR)

    # register communitorc for embedding layer.
    for embedding in _submodule_filter(model, Embedding1D):
        _embedding_communicator.register_module_hook(embedding)

    # register communictor for head layer.
    ScaleColumnParallelLinear.register_cls_communicator(_head_communicator)
    RewardModelLinear.register_cls_communicator(_head_communicator)

    return isp_communicator_wrapper


@llm_timeout(func_name="initialize_optimizer")
def initialize_optimizer(model: Union[nn.Module, nn.ModuleList], isp_communicator: ISPCommunicatorWrapper = None):
    """
    Initialize optimizer.

    Args:
        model (:class:`torch.nn.Module`): Your model instance to be trained or evaluated.

    Returns:
        A tuple of (optimizer, beta2_scheduler, lr_scheduler).
    """

    adam_cfg = gpc.config.adam
    zero_cfg = gpc.config.hybrid_zero_optimizer
    grad_scal_cfg = gpc.config.grad_scaler
    use_apex_adam = getattr(gpc.config, "use_apex_adam", False)

    if "use_split_tensor_optim" in zero_cfg and zero_cfg.use_split_tensor_optim:
        map_param_block(model)

    params = create_param_groups(model, adam_cfg.weight_decay)

    naive_optimizer = new_compatible_adamw(
        params=params,
        lr=adam_cfg.lr,
        betas=(adam_cfg.adam_beta1, adam_cfg.adam_beta2),
        eps=adam_cfg.adam_eps,
        use_apex_adam=use_apex_adam,
    )

    if (
        zero_cfg.overlap_sync_grad
        and gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
        and gpc.is_pipeline_first_stage() is False
    ):
        # When pipeline parallelism is enabled, we prefer to only enable optimizer
        # gradient communication overlap in the first stage, to avoid amplifying
        # the communication overhead stage by stage in cases where the optimizer
        # communication overhead is greater than the compute overhead.
        # For pipeline stages except the first, even if overlap is not enabled,
        # their gradient synchronization overhead can be well hidden by
        # the inherent bubbles of pipeline parallelism.
        zero_cfg.overlap_sync_grad = False

    if zero_cfg.overlap_sync_param:
        param_bcast_sync_handler = ParamAsyncBcastHandler(ParallelMode.ZERO1, model, isp_communicator)
    else:
        param_bcast_sync_handler = None

    if not gpc.config.parallel.zero1.fsdp:
        if (
            "use_split_tensor_optim" not in gpc.config.hybrid_zero_optimizer
            or not gpc.config.hybrid_zero_optimizer.use_split_tensor_optim
        ):
            optimizer = HybridZeroOptimizer(
                naive_optimizer,
                grad_scal_cfg=grad_scal_cfg,
                zero_cfg=zero_cfg,
                param_bcast_sync_handler=param_bcast_sync_handler,
                isp_communicator=isp_communicator,
            )
        else:
            optimizer = HybridZeroOptimizer_v2(
                naive_optimizer,
                grad_scal_cfg=grad_scal_cfg,
                zero_cfg=zero_cfg,
                param_bcast_sync_handler=param_bcast_sync_handler,
                isp_communicator=isp_communicator,
            )
    else:
        optimizer = FSDPadaptOptimizer(
            naive_optimizer,
            grad_scal_cfg=grad_scal_cfg,
            zero_cfg=zero_cfg,
        )

    beta2_scheduler = Beta2Scheduler(optimizer=naive_optimizer, **gpc.config.beta2_scheduler)

    lr_scheduler = FineTuneCosineAnnealingWarmupLR(optimizer, **gpc.config.lr_scheduler)

    return optimizer, beta2_scheduler, lr_scheduler


def get_scheduler_hooks(metric, zero_optim, isp_communicator_wrapper) -> List[SchedulerHook]:
    scheduler_hooks: List[SchedulerHook] = []

    if metric is not None:
        scheduler_hooks.append(
            SchedulerMetricHook(
                metric=metric,
                skip=(
                    gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
                    and hasattr(gpc.config.model, "num_chunks")
                    and gpc.config.model.num_chunks > 1
                    and gpc.config.parallel["pipeline"].get("interleaved_overlap", False)
                ),
            ),
        )

    if isp_communicator_wrapper is not None:
        for isp_communicator in isp_communicator_wrapper.isp_communicators:
            if isp_communicator is not None and isp_communicator.overlap:
                scheduler_hooks.append(ISPCommunicatorSchedulerHook(isp_communicator, zero_optim))

    return scheduler_hooks


@llm_timeout(func_name="load_new_batch")
def load_new_batch(train_dl: DataLoader, train_iter: Iterable, train_state: TrainState):
    """
    Load and return the new batch data based on training data loader.

    Args:
        train_dl (torch.utils.data.DataLoader): Dataloader for training.
        train_iter (Iterable): Data iterator from which get a batch of data, obtained by calling iter(dataloader).
        train_state (TrainState): Current training state.

    Returns: A batch data and the updated train_iter.
    """

    timer("batch-gen").start()
    try:
        batch = next(train_iter)  # structure is ({'input_ids': Tensor, 'cu_seqlens': Tensor}, Tensor)
        if hasattr(train_state, "batch_sampler_iter"):
            next(train_state.batch_sampler_iter)
    except StopIteration:
        train_iter = iter(train_dl)
        batch = next(train_iter)
        train_state.num_consumed_samples_in_epoch = 0
        if hasattr(train_state, "batch_sampler"):
            train_state.batch_sampler.batch_count = 0
            train_state.batch_sampler.num_consumed_samples_in_epoch = 0
            train_state.batch_sampler_iter = iter(train_state.batch_sampler)
            next(train_state.batch_sampler_iter)
    timer("batch-gen").stop()

    if batch[0].get("type_ids", None) is not None:
        # if use_packed_dataset is False, we need to unpack type_ids
        if not gpc.config.data.use_packed_dataset:
            batch[0]["type_ids"] = unpack_type_ids(batch[0]["type_ids"], batch[0]["cu_seqlens"])

    return batch, train_iter


def initialize_llm_profile(profiling: bool = False, start_time: str = None):
    """Initialize and return the profiler context manager instance."""

    if profiling and gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        schedule_config = {"wait": 1, "warmup": 1, "active": 1, "repeat": 1, "skip_first": 3}
        trace_path = (
            f"RUN/{gpc.config.JOB_NAME}/{start_time}/traces/rank{gpc.get_global_rank()}_"
            f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            f"wp{gpc.get_local_rank(ParallelMode.WEIGHT)}_"
            f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}"
        )
        if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
            )
            llm_profile = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(**schedule_config),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            )
            logger.info(f"Do profiling for NPU on rank {gpc.get_global_rank()}!")
        else:
            llm_profile = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(**schedule_config),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
                with_stack=True,
                with_modules=True,
                profile_memory=True,
            )
            logger.info(f"Do profiling for GPU on rank {gpc.get_global_rank()}!")
    else:
        llm_profile = DummyProfile()

    return llm_profile


@llm_timeout(func_name="record_current_batch_training_metrics")
def record_current_batch_training_metrics(
    get_tflops_func,
    logger,
    writer,
    success_update,
    batch_count,
    batch,
    train_state,
    optimizer,
    beta2_scheduler,
    engine,
    start_time,
    very_begining_time,
    loss,
    moe_loss,
    grad_norm,
    metric,
):
    """
    Print some training metrics of current batch.
    """

    set_env_var(key="LAST_ACTIVE_TIMESTAMP", value=int(time.time()))

    timer.store_last_timers()
    success_update = True
    if success_update in (0, True):
        train_state.num_consumed_tokens += batch[1].nelement() * gpc.get_world_size(ParallelMode.DATA)
    if gpc.is_no_pp_or_last_stage():
        acc_perplex = metric.get_metric()

    if success_update and gpc.is_rank_for_log():
        lr = optimizer.param_groups[0]["lr"]
        if hasattr(engine.optimizer, "grad_scaler"):
            scaler = engine.optimizer.grad_scaler._scale.item()
        elif hasattr(engine.optimizer.optim, "grad_scaler"):
            scaler = engine.optimizer.optim.grad_scaler._scale.item()

        num_tokens_in_batch = batch[1].nelement()
        real_num_tokens = math.ceil(acc_perplex.pop("real_token_num") / gpc.get_world_size(ParallelMode.GLOBAL))
        num_samples_in_batch = sum([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        max_length_in_batch = max([(b[1:] - b[:-1]).max().item() for b in batch[0]["cu_seqlens"]])
        max_samples_in_batch = max([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        min_samples_in_batch = min([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        time_cost = time.time() - start_time
        tk_per_gpu = round(
            num_tokens_in_batch * gpc.get_world_size(ParallelMode.DATA) / gpc.get_world_size(ParallelMode.GLOBAL),
            4,
        )
        tgs_statistic = train_state.tgs_statistic
        tgs_statistic["sum_step"] += 1
        tgs_statistic["sum_tg"] += tk_per_gpu
        tgs_statistic["total_time"] = time.time() - very_begining_time
        tgs_statistic["sum_last_tg_10"] += tk_per_gpu
        tgs_statistic["sum_last_time_10"] += time_cost
        tgs_statistic["sum_last_tg_50"] += tk_per_gpu
        tgs_statistic["sum_last_time_50"] += time_cost
        tgs_statistic["SMA_tg_50"] += tk_per_gpu
        tgs_statistic["SMA_time_50"] += time_cost
        tgs_statistic["SMA_tg_50_list"].append(tk_per_gpu)
        tgs_statistic["SMA_time_50_list"].append(time_cost)
        if tgs_statistic["sum_step"] > 50:
            tgs_statistic["SMA_tg_50"] -= tgs_statistic["SMA_tg_50_list"][0]
            tgs_statistic["SMA_time_50"] -= tgs_statistic["SMA_time_50_list"][0]
            tgs_statistic["SMA_tg_50_list"].popleft()
            tgs_statistic["SMA_time_50_list"].popleft()

        last_tgs_1 = round(tk_per_gpu / time_cost, 2)
        tgs_statistic["sum_tgs"] += last_tgs_1

        if tgs_statistic["sum_step"] % 10 == 0:
            tgs_statistic["last_tgs_10"] = round(tgs_statistic["sum_last_tg_10"] / tgs_statistic["sum_last_time_10"], 2)
            tgs_statistic["sum_last_tg_10"] = 0
            tgs_statistic["sum_last_time_10"] = 0

        if tgs_statistic["sum_step"] % 50 == 0:
            tgs_statistic["last_tgs_50"] = round(tgs_statistic["sum_last_tg_50"] / tgs_statistic["sum_last_time_50"], 2)
            tgs_statistic["sum_last_tg_50"] = 0
            tgs_statistic["sum_last_time_50"] = 0

        last_tgs_10 = tgs_statistic["last_tgs_10"]
        last_tgs_50 = tgs_statistic["last_tgs_50"]

        tgs_all = round(tgs_statistic["sum_tg"] / tgs_statistic["total_time"], 2)
        tgs_avg = round(tgs_statistic["sum_tgs"] / tgs_statistic["sum_step"], 2)
        tgs_SMA = round(tgs_statistic["SMA_tg_50"] / tgs_statistic["SMA_time_50"], 2)
        log_file = os.environ.get("LOG_FILE", None)
        model_size = os.environ.get("MODEL_SIZE", None)
        method = os.environ.get("METHOD", None)
        seqlen = os.environ.get("SEQ_LEN", None)
        cp_size = os.environ.get("CP_SIZE", None)
        hp_size = os.environ.get("HP_SIZE", None)
        log_str = f"model_size={model_size} method={method} seqlen={seqlen} cp_size={cp_size} hp_size={hp_size}"
        with open(log_file, "a") as f:
            f.write(f"{log_str} avg_tgs={tgs_all}\n")
        tflops = get_tflops_func(time_cost)

        tgs_origin = round(
            num_tokens_in_batch
            * gpc.get_world_size(ParallelMode.DATA)
            / gpc.get_world_size(ParallelMode.GLOBAL)
            / time_cost,
            2,
        )

        real_tgs = round(
            real_num_tokens / time_cost,
            2,
        )

        infos = {
            "tflops": tflops,
            "step": batch_count,
            "loss": loss.item() - moe_loss.item() if moe_loss is not None else loss.item(),
            "real_tgs": real_tgs,
            "tgs (tokens/gpu/second)": tgs_origin,
            "tgs/last_tgs_1": last_tgs_1,
            "tgs/tgs_all": tgs_all,
            "tgs/tgs_avg": tgs_avg,
            "tgs/tgs_SMA": tgs_SMA,
            "tgs/last_tgs_10": last_tgs_10,
            "tgs/last_tgs_50": last_tgs_50,
            "lr": lr,
            "loss_scale": scaler,
            "grad_norm": grad_norm,
        }
        if moe_loss is not None:
            infos["moe_loss"] = moe_loss.item()

        infos["micro_num"] = len(batch[1])
        infos["num_consumed_tokens"] = train_state.num_consumed_tokens
        infos["inf_nan_skip_batches"] = train_state.inf_nan_skip_batches
        infos["num_samples_in_batch"] = num_samples_in_batch  # the number of batches which have the most samples
        infos["largest_length"] = max_length_in_batch  # the longest input
        infos["largest_batch"] = max_samples_in_batch  # the batch with the most samples
        infos["smallest_batch"] = min_samples_in_batch
        infos["adam_beta2"] = beta2_scheduler.get_beta2()

        fwd_bwd_time = round(timer("fwd-bwd").elapsed(), 2)
        infos["fwd_bwd_time"] = fwd_bwd_time
        bwd_time = round(timer("bwd").elapsed(), 2)
        infos["bwd_time"] = bwd_time

        for key, value in acc_perplex.items():
            infos[key] = value

        line = ""
        for key, value in infos.items():
            line += f"{key}={value} "
            if isinstance(value, dict):
                writer.add_scalars(key=key, value=value, step=train_state.step_count)
            else:
                writer.add_scalar(key=key, value=value, step=train_state.step_count)

        logger.info(line)

        # if loss spike occurs, send alert info to feishu
        mm.monitor_loss_spike(
            alert_address=gpc.config.monitor.alert.feishu_alert_address,
            step_count=batch_count,
            cur_step_loss=loss.item(),
        )


def inject_embed(model: nn.Module, inject=False, interactive=False) -> None:
    def traverse(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Embedding) and not isinstance(child, Embedding1D):
                msg = (
                    f"To get parallel training enabled, module {name} of type {nn.Embedding.__name__} "
                    + f"is required to be replaced with {Embedding1D.__name__}."
                )
                if inject:
                    help_msg = f"Do you want to replace {name}? (y/n)"
                    opt = timeout_input(
                        f"{msg}\n{help_msg}",
                        default="y",
                        timeout=60,
                        interactive=interactive,
                    )
                    if opt in ["y", "yes"]:
                        child_new = Embedding1D(
                            num_embeddings=child.num_embeddings,
                            embedding_dim=child.embedding_dim,
                            padding_idx=child.padding_idx,
                        ).to(device=child.weight.device, dtype=child.weight.dtype)
                        setattr(module, name, child_new)
                    else:
                        if gpc.is_rank_for_log():
                            logger.warning(f"Skip replacing {name}")
                else:
                    if gpc.is_rank_for_log():
                        logger.warning(msg)
            else:
                traverse(child)

    traverse(model)


def inject_linear(model: nn.Module, inject=False, interactive=False) -> None:
    def traverse(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and not isinstance(child, ParallelLinearWithCommExt):
                msg = (
                    f"To get parallel training enabled, module {name} of type {nn.Linear.__name__} "
                    + f"is required to be replaced with {new_linear.__name__}."
                )
                if inject:
                    help_msg = f"Do you want to replace {name}? (y/n)"
                    opt = timeout_input(
                        f"{msg}\n{help_msg}",
                        default="y",
                        timeout=60,
                        interactive=interactive,
                    )
                    if opt in ["y", "yes"]:
                        child_new = new_linear(
                            name=LINEAR2NEWLINEAR_NAME_MAPPING.get(name, name),
                            in_features=child.in_features,
                            out_features=child.out_features,
                            bias=child.bias is not None,
                        ).to(device=child.weight.device, dtype=child.weight.dtype)
                        setattr(module, name, child_new)
                    else:
                        if gpc.is_rank_for_log():
                            logger.warning(f"Skip replacing {name}")
                else:
                    if gpc.is_rank_for_log():
                        logger.warning(msg)
            else:
                traverse(child)

    traverse(model)


def inject_norm(model: nn.Module, inject=False, interactive=False) -> None:
    def traverse(module):
        for name, child in module.named_children():
            cls_name = type(child).__name__
            if "RMSNorm" in cls_name:
                msg = (
                    f"To re-use unified RMSNorm implementation, {cls_name} "
                    + f"is suggested to be replaced with {new_layer_norm.__name__}."
                )
                if inject:
                    help_msg = f"Do you want to replace {name}? (y/n)"
                    opt = timeout_input(
                        f"{msg}\n{help_msg}",
                        default="y",
                        timeout=60,
                        interactive=interactive,
                    )
                    if opt in ["y", "yes"]:
                        child_new = new_layer_norm(
                            norm_type="rmsnorm",
                            normalized_shape=child.weight.shape,
                            eps=child.variance_epsilon,
                        ).to(device=child.weight.device, dtype=child.weight.dtype)
                        setattr(module, name, child_new)
                    else:
                        if gpc.is_rank_for_log():
                            logger.warning(f"Skip replacing {name}")
                else:
                    if gpc.is_rank_for_log():
                        logger.warning(msg)
            else:
                traverse(child)

    traverse(model)


def inject_config(model: nn.Module) -> None:
    if hasattr(model.config, "text_config"):
        model_config = model.config.text_config
    else:
        model_config = model.config
    gpc.config.model.vocab_size = gpc.config.VOCAB_SIZE = model_config.vocab_size
    gpc.config.model.hidden_size = gpc.config.HIDDEN_SIZE = model_config.hidden_size
    gpc.config.model.num_layers = gpc.config.NUM_LAYER = model_config.num_hidden_layers
    gpc.config.model.num_attention_heads = gpc.config.NUM_ATTENTION_HEAD = model_config.num_attention_heads
    gpc.config.model.mlp_ratio = gpc.config.MLP_RATIO = model_config.intermediate_size / model_config.hidden_size
    # For models that use GQA
    if hasattr(model_config, "num_key_value_heads"):
        gpc.config.model.num_kv_attention_heads = gpc.config.NUM_KV_ATTENTION_HEAD = model_config.num_key_value_heads


def inject_model_helper(model: Union[nn.Module, nn.ModuleList], inject_info: Optional[Dict] = None) -> None:
    # get inject_info
    if inject_info is not None:
        inject = inject_info.get("inject", False)
        interactive = inject_info.get("interactive", False)
        modules = inject_info.get("modules", [])
        reset_params = inject_info.get("reset_params", False)
        extra_linear2newlinear = inject_info.get("extra_linear2newlinear", {})
    else:
        inject = False
        interactive = False
        modules = []
        reset_params = False
        extra_linear2newlinear = {}

    LINEAR2NEWLINEAR_NAME_MAPPING.update(extra_linear2newlinear)

    inject_funcs = {
        "embed": inject_embed,
        "linear": inject_linear,
        "norm": inject_norm,
    }

    if not isinstance(model, nn.ModuleList):
        model = [model]

    # inject modules
    for _chunk in model:
        if gpc.get_world_size(ParallelMode.DATA) == gpc.get_world_size(ParallelMode.GLOBAL) and gpc.get_world_size(
            ParallelMode.WEIGHT_DATA
        ) == gpc.get_world_size(ParallelMode.GLOBAL):
            continue
        for mod in modules:
            inject_funcs[mod](_chunk, inject, interactive)

    # reset parameters and move model to device
    for _chunk in model:
        if inject:
            if reset_params:
                _chunk.reset_parameters()
            _chunk.to(get_current_device())

    # inject configs
    if inject:
        inject_config(model[0])
        if gpc.is_rank_for_log():
            logger.info(
                f"inject is enabled, please check the model carefully, "
                f"if there are any problems, please report issue to us. "
                f"The injected model is \n {model}"
            )
