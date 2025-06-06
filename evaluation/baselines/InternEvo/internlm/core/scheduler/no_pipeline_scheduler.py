#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

from typing import Any, Callable, Iterable, List, Optional

import torch
import torch.distributed as dist

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.engine import Engine
from internlm.utils.common import (
    SchedulerHook,
    check_data_is_packed,
    conditional_context,
    get_current_device,
)
from internlm.utils.logger import get_logger
from internlm.utils.timeout import llm_timeout

from .base_scheduler import BaseScheduler

logger = get_logger(__file__)


class NonPipelineScheduler(BaseScheduler):
    """A helper schedule class for no pipeline parallelism running environment.
    During one process, it loads a batch of dataset and feeds it to the model.
    After getting the output and calculating the loss, it will use :meth:`step`
    to update the parameters if it is in training mode.

    Args:
        data_process_func (Callable, optional): The preprocessing function which receives a batch of data
            and returns a tuple in the form of (data, label), and it will be executed in load_batch.
        gradient_accumulation_steps(int, optional): the steps of gradient accumulation, 1 for disable
            gradient accumulation.

    Examples:
        >>> # this shows an example of customized data_process_func
        >>> def data_process_func(dataloader_output):
        >>>     item1, item2, item3 = dataloader_output
        >>>     data = (item1, item2)
        >>>     label = item3
        >>>     return data, label
    """

    def __init__(
        self,
        data_process_func: Callable = None,
        gradient_accumulation_size: int = 1,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
    ):
        self._grad_accum_size = gradient_accumulation_size
        self._grad_accum_offset = 0

        self._hooks = scheduler_hooks

        super().__init__(data_process_func)

    def pre_processing(self, engine: Engine):
        """Performs actions before running the schedule.

        Args:
           engine (internlm.core.Engine): InternLM engine for training and inference.
        """
        pass

    def _call_hooks(self, func_name: str, *args, **kwargs) -> None:
        for hook in self._hooks:
            getattr(hook, func_name)(self, *args, **kwargs)

    def _load_accum_batch(self, data: Any, label: Any):
        """Loads a batch of data and label for gradient accumulation.

        Args:
            data (Any): The data to be loaded.
            label (Any): The label to be loaded.
        """

        _data, _label = self._load_micro_batch(
            data=data, label=label, offset=self._grad_accum_offset, bsz_stride=self._bsz_stride
        )
        self._grad_accum_offset += self._bsz_stride

        if self.data_process_func:
            _data, _label = self.data_process_func(_data, _label)

        return _data, _label

    def _train_one_batch(
        self,
        data: Any,
        label: Any,
        engine: Engine,
        forward_only: bool = False,
        return_loss: bool = True,
        return_output: bool = False,
        scale_loss: int = 1,
    ):
        """Trains one batch of data.

        Args:
            data (Any): The data to be trained.
            label (Any): The label for the data.
            engine (internlm.core.Engine): InternLM engine for training and inference.
            forward_only (bool, optional): If True, the model is run for the forward pass, else back propagation will
                be executed.
            return_loss (bool, optional): Loss will be returned if True.
            return_output (bool, optional): Output will be returned if True.
            scale_loss (int, optional): The scale factor for the loss.
        """

        # forward
        with conditional_context(torch.no_grad(), enable=forward_only):
            self._call_hooks("before_forward", data)
            if hasattr(gpc.config.model, "num_experts"):
                # moe is used
                output, moe_losses = self._call_engine(engine, data)
            else:
                output = self._call_engine(engine, data)
            self._call_hooks("after_forward", output)

            # self._call_hooks("post_helper_func", output, label)

            if return_loss:
                self._call_hooks("before_criterion", output, label)
                loss = self._call_engine_criterion(engine, output, label)
                self._call_hooks("after_criterion", loss)
                moe_loss = (
                    sum(moe_losses) * gpc.config.loss.moe_loss_coeff  # pylint: disable=E0606
                    if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1
                    else torch.tensor(0.0, device=get_current_device(), dtype=gpc.config.model.get("dtype"))
                )
                # the moe_loss is computed among the "tensor" group if sequence parallel is enabled,
                # so we need to do allreduce
                if gpc.config.parallel.sequence_parallel or gpc.config.parallel.expert.no_tp:
                    dist.all_reduce(moe_loss, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.TENSOR))
                    moe_loss.div_(gpc.get_world_size(ParallelMode.TENSOR))
                moe_loss /= scale_loss
                loss /= scale_loss
                loss += moe_loss

        # clear output before backward for releasing memory resource
        if not return_output:
            output = None

        # backward
        if not forward_only:
            self._call_hooks("before_backward", None, None)
            engine.backward(loss)
            self._call_hooks("after_backward", None)

        if not return_loss:
            loss, moe_loss = None, None

        return output, loss, moe_loss

    @llm_timeout(func_name="nopp_forward_backward_step")
    def forward_backward_step(
        self,
        engine: Engine,
        data_iter: Iterable,
        forward_only: bool = False,
        return_loss: bool = True,
        return_output_label: bool = True,
    ):
        """The process function that loads a batch of dataset and feeds it to the model.
        The returned labels and loss will None if :attr:`return_loss` is False.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                If True, the model is run for the forward pass, else back propagation will be executed.
            return_loss (bool, optional): Loss will be returned if True.
            return_output_label (bool, optional): Output and label will be returned if True.
        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
        """
        assert (
            forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        # actual_batch_size is micro_num when training,
        # actual_batch_size is micro_num * micro_bsz when evaluating
        batch_data, actual_batch_size = engine.load_batch(data_iter)

        if check_data_is_packed(batch_data):
            micro_num = actual_batch_size
        else:
            micro_num = actual_batch_size // gpc.config.data["micro_bsz"]

        self._grad_accum_size = micro_num  # Rampup or variable bsz size.
        self._bsz_stride = actual_batch_size // self._grad_accum_size

        data, label = batch_data

        loss = 0 if return_loss else None
        moe_loss = 0 if return_loss else None
        outputs = []
        labels = []

        # reset accumulation microbatch offset
        self._grad_accum_offset = 0

        for _current_accum_step in range(self._grad_accum_size):
            if engine.optimizer is not None:
                if _current_accum_step == self._grad_accum_size - 1:
                    engine.optimizer.skip_grad_reduce = False
                else:
                    engine.optimizer.skip_grad_reduce = True

            _data, _label = self._load_accum_batch(data, label)

            _output, _loss, _moe_loss = self._train_one_batch(
                _data, _label, engine, forward_only, return_loss, return_output_label, self._grad_accum_size
            )

            if return_loss:
                loss += _loss
                moe_loss += _moe_loss

            if return_output_label:
                outputs.append(_output)
                labels.append(_label)

        if not return_output_label:
            outputs, labels = None, None

        # Compatible for non-moe
        if hasattr(gpc.config.model, "num_experts"):
            return outputs, labels, loss, moe_loss
        else:
            return outputs, labels, loss
