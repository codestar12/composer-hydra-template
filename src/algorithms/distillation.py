from typing import Optional, Union, Tuple, Callable

from composer import Algorithm, Event, State
from composer.loggers import Logger

from composer.models import ComposerModel
import torch.nn as nn
import torch

from .distillation_kl_loss import DistillKL

class Distillation(Algorithm):
    """_summary_

    Args:
        teacher (nn.Module) : a teacher model 
        kd_loss_fn (Callable): loss function to perform distillation with
        kd_loss_weight (float): weighting of kd loss
        org_loss_weight (float): weighting of original loss
        input_key (str, int, tuple): key of input values from batch
        teacher_weights_pth (str): path to teacher weights
    """

    def __init__(
            self, 
            teacher: nn.Module, 
            kd_loss_fn: nn.Module = DistillKL(),
            kd_loss_weight: float = 0.9, 
            org_loss_weight: float = 0.1,
            input_key: Union[str, int] = 0,
            teacher_weights_pth: str = None,
        ):
        super().__init__()
        self.teacher = teacher
        if teacher_weights_pth is not None:
            ckpt = torch.load(teacher_weights_pth)
            weights = None
            try:
                weights = ckpt['state']['model']
            except KeyError:
                print("key state/model not found. Only composer models supported currently")
                raise
            try:
                self.teacher.load_state_dict(weights)
            except:
                print("error loading teacher model")
                raise
        self.kd_loss_fn = kd_loss_fn
        self.kd_loss_weight = kd_loss_weight
        self.org_loss_weight = org_loss_weight
        self.input_key = input_key

    def match(self, event: Event, state: State) -> bool:
        return event == Event.FIT_START or event == Event.AFTER_LOSS 

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        input  = None
        if event == Event.FIT_START:
            # move teacher to correct device after init
            self._move_teacher_model_to_device(self.teacher.module, state.model)
        elif event == Event.AFTER_LOSS:
            input = state.batch_get_item(self.input_key) 

            with torch.no_grad(): 
                t_output = self.teacher.module(input)

            base_loss = state.loss
            s_output = state.outputs
            kd_loss = self.kd_loss_fn(t_output, s_output)

            if type(base_loss) is tuple:
                state.loss = tuple([self.org_loss * v for v in state.loss] + [kd_loss])
            elif type(base_loss) is dict:
                new_loss = dict()
                for k, v in base_loss:
                    new_loss[k] = self.org_loss_weight * v
                
                new_loss["kd_loss"] = self.kd_loss_weight * kd_loss 

                state.loss = new_loss 
            else:
                state.loss = self.org_loss_weight * base_loss + self.kd_loss_weight * kd_loss 
    
    def _move_teacher_model_to_device(self, teacher_model: Union[ComposerModel, nn.Module], destination_model: Union[ComposerModel, nn.Module]):
        """Ensures the tensors of a teacher model are on the same device as a destination model."""
        with torch.no_grad():
            #device = next(destination_model.parameters()).device
            if torch.cuda.is_available():
                self.teacher.to(torch.cuda.current_device())
            # destination_params = destination_model.parameters()
            # teacher_params = teacher_model.parameters()
            # teacher_model.param_list = [s.to(d.device) for s, d in zip(teacher_params, destination_params)]

            # destination_buffers = destination_model.buffers()
            # teacher_buffers = teacher_model.buffers()
            # teacher_model.buffer_list = [s.to(d.device) for s, d in zip(teacher_buffers, destination_buffers)]


