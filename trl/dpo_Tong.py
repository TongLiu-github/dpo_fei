
import sys
import time
import torch
import logging
import warnings
import transformers
from enum import Enum
from datasets import load_dataset
from accelerate import PartialState
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer

# trl
from trl.commands.cli_utils import DPOScriptArguments, TrlParser
from trl import (
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    maybe_extract_prompt,
    maybe_apply_chat_template,
)
from trl0.trl.trainer.dpo_trainer import DPOTrainer
from trl0.trl.trainer.dpo_config import DPOConfig



# # from sampo
# llama3_template = "<|start_header_id|>user<|end_header_id|>\n\n{PROMPT}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
# tulu2_template = "<|user|>\n{PROMPT}\n<|assistant|>\n"
#
# # from simpo
# MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
#
# but finally, I decided to use automatic template from trl

@dataclass
class DPOScriptArguments0:
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_train_split: str = field(default="train", metadata={"help": "The dataset split to use for training"})
    dataset_test_split: str = field(default="test", metadata={"help": "The dataset split to use for evaluation"})
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "debug argument for distributed training;"
            "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"},
    )
    train_data_path: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "The input evaluation data file (a jsonlines)."})



class FDivergenceType(Enum):
    REVERSE_KL = "reverse_kl"
    JS_DIVERGENCE = "js_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"


class FDivergenceConstants:
    ALPHA_DIVERGENCE_COEF_KEY = "alpha_divergence_coef"
    ALPHA_DIVERGENCE_COEF_DEFAULT = 1.0


@dataclass
class DPOConfig0(TrainingArguments):
    r"""
    Configuration class for the [`DPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        beta (`float`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model. For the IPO loss (`loss_type="ipo"`), β is the regularization parameter denoted by τ in
            the [paper](https://huggingface.co/papers/2310.12036).
        label_smoothing (`float`, *optional*, defaults to `0.0`):
            Robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report and
            [Robust DPO](https://huggingface.co/papers/2403.00409) paper that should be between `0.0` and `0.5`.
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            Type of loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"hinge"`: hinge loss on the normalized likelihood from the [SLiC](https://huggingface.co/papers/2305.10425) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.
                - `"exo_pair"`: pairwise EXO loss from the [EXO](https://huggingface.co/papers/2402.00856) paper.
                - `"nca_pair"`: pairwise NCA loss from the [NCA](https://huggingface.co/papers/2402.05369) paper.
                - `"robust"`: unbiased estimate of the DPO loss that is robust to preference noise from the [Robust DPO](https://huggingface.co/papers/2403.00409) paper.
                - `"bco_pair"`: pairwise BCO loss from the [BCO](https://huggingface.co/papers/2404.04656) paper.
                - `"sppo_hard"`: SPPO loss with hard label from the [SPPO](https://huggingface.co/papers/2405.00675) paper.
                - `"aot"`: AOT loss for paired datasets from the [AOT](https://huggingface.co/papers/2406.05882) paper.
                - `"aot_pair"`: AOT loss for unpaired datasets from the [AOT](https://huggingface.co/papers/2406.05882) paper.
                - `"apo_zero"`: APO-zero loss from the [APO](https://huggingface.co/papers/2408.06266) paper.
                - `"apo_down"`: APO-down loss from the [APO](https://huggingface.co/papers/2408.06266) paper.

        label_pad_token_id (`int`, *optional*, defaults to `-100`):
            Label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`Optional[int]`, *optional*, defaults to `None`):
            Padding value to use. If `None`, the padding value of the tokenizer is used.
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            Truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the
            default data collator.
        max_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the sequences (prompt + completion) in the batch. This argument is required if you want
            to use the default data collator.
        max_prompt_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_completion_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the target. This argument is required if you want to use the default data collator and
            your model is an encoder-decoder.
        is_encoder_decoder(`Optional[int]`, *optional*, defaults to `None`):
            When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument,
            you need to specify if the model returned by the callable is an encoder-decoder model.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.
        generate_during_eval (`bool`, *optional*, defaults to `False`):
            Truncation mode to use when the prompt is too long. Possible values are `"keep_end"` or `"keep_start"`.
            This argument is required if you want to use the default data collator.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Whether to precompute reference model log probabilities for training and evaluation datasets. This is
            useful when training without the reference model to reduce the total GPU memory needed.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        model_init_kwargs (`Optional[Dict[str, Any]]`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
        ref_model_init_kwargs (`Optional[Dict[str, Any]]`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the reference model
            from a string.
        model_adapter_name (`Optional[str]`, *optional*, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`Optional[str]`, *optional*, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        reference_free (`bool`, *optional*, defaults to `False`):
            If `True`, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal
            probability to all responses.
        force_use_ref_model (`bool`, *optional*, defaults to `False`):
            In case one passes a PEFT model for the active model and you want to use a different model for the
            ref_model, set this flag to `True`.
        f_divergence_type (`str`, *optional*, defaults to `FDivergenceType.REVERSE_KL`):
            Type of f-divergence regularization function to compute divergence between policy and reference model.
        f_alpha_divergence_coef (`float`, *optional*, defaults to `1.0`):
            α coefficient in the α-divergence \\(u^{-\\alpha}\\) regularization function for DPO loss.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            When set to `True`, the reference model is synchronized with the active model every `ref_model_sync_steps`
            steps, using the `ref_model_mixup_alpha` parameter. This synchronization originites from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.9`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`
            To use this parameter, you must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `64`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.
        rpo_alpha (`float`, *optional*, defaults to `None`):
            α parameter from the [RPO](https://huggingface.co/papers/2404.19733) paper (v3), which controls the
            weighting of the NLL term in the loss. If `None`, no weighting is applied and the loss is the same as the
            DPO loss. The paper recommends `rpo_alpha=1.0`.
    """

    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: Literal[
        "dpo",
        "hinge",
        "ipo",
        "exo_pair",
        "nca_pair",
        "robust",
        "bco_pair",
        "sppo_hard",
        "aot",
        "aot_pair",
        "apo_zero",
        "apo_down",
        "focal"
    ] = "dpo"
    label_pad_token_id: int = -100
    padding_value: Optional[int] = None
    truncation_mode: str = "keep_end"
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_target_length: Optional[int] = None  # deprecated in favor of max_completion_length
    max_completion_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = False
    precompute_ref_log_probs: bool = False
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[Dict[str, Any]] = None
    ref_model_init_kwargs: Optional[Dict[str, Any]] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    f_divergence_type: FDivergenceType = FDivergenceType.REVERSE_KL
    f_alpha_divergence_coef: float = 1.0
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    rpo_alpha: Optional[float] = None
    report_to: Optional[str] = None

    def __post_init__(self):
        if self.max_target_length is not None:
            warnings.warn(
                "The `max_target_length` argument is deprecated in favor of `max_completion_length` and will be removed in a future version.",
                FutureWarning,
            )
            if self.max_completion_length is None:
                self.max_completion_length = self.max_target_length

        return super().__post_init__()



if __name__ == "__main__":

    print("1. read arguments")
    parser = TrlParser((DPOScriptArguments0, DPOConfig0, ModelConfig))
    data_args, training_args, model_config = parser.parse_args_and_config()

    # other args
    # bf16 = True, bf16_full_eval=False
    training_args.bf16 = True
    TIMER = time.gmtime()
    training_args.run_name = "{}_{}_{}_{}_{}_{}_{}".format(
        model_config.model_name_or_path.split("/")[-1],
        data_args.dataset_name.split("/")[-1],
        training_args.loss_type,
        training_args.beta,
        TIMER.tm_mday,
        TIMER.tm_hour,
        TIMER.tm_min
    )


    # set logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_config}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Model & Tokenizer
    ###################
    print("2. read model & tokenizer")
    # trl dpo.py的torch_dtype, quantization_config, model_kwargs, model这几个也和simpo的保持了一致
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    import os
    os.environ['CURL_CA_BUNDLE'] = ''

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )

    # the following is from trl dpo.py:
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if data_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    print("3. read dataset")
    if data_args.train_data_path != None:
        data_files = {}
        data_files["train"] = data_args.train_data_path
        data_files["valid"] = data_args.eval_data_path
        dataset = load_dataset(
            "json",
            data_files=data_files
        )
    else:
        dataset = load_dataset(data_args.dataset_name)

        # note that we cannot process "HuggingFaceH4/ultrafeedback_binarized" using the following code
        with PartialState().local_main_process_first():
            dataset = dataset.map(maybe_extract_prompt, num_proc=training_args.dataset_num_proc)
            dataset = dataset.map(
                maybe_apply_chat_template, num_proc=training_args.dataset_num_proc, fn_kwargs={"tokenizer": tokenizer}
            )

    ##########
    # Training
    ################
    print("4. training")
    # if "wandb" in training_args.report_to:
    #     import wandb
    #     # note here, it's "name", nor "project" name.
    #     wandb.init(name = training_args.run_name)

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[data_args.dataset_train_split],
        eval_dataset=dataset[data_args.dataset_test_split],
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_model(training_args.output_dir)
