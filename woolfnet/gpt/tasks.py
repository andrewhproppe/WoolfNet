import logging
from contextlib import nullcontext
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import mlflow
import optax
import torch
from flax import nnx
from tokenizers import ByteLevelBPETokenizer

from woolfnet.config import load_yaml
from woolfnet.data.dataset import CorpusDataset
from woolfnet.gpt import GPT_ARTIFACTS, LR_SCHEDULER_REGISTRY, LR_SCHEDULER_REGISTRY_JAX
from woolfnet.gpt.config import GPTConfig
from woolfnet.gpt.metrics import cross_entropy_loss
from woolfnet.gpt.model import GPT as GPTTorch
from woolfnet.gpt.model_jax import GPT as GPTJax
from woolfnet.gpt.model_nnx import GPT as GPTNNX
from woolfnet.gpt.training import (
    gpt_finetune_loop,
    gpt_training_loop,
    gpt_training_loop_jax,
    gpt_training_loop_nnx,
)
from woolfnet.utils.mlflow import configure_mlflow, make_versioned_model_name

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-config",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the GPT model config yaml",
)
@click.option(
    "--training-config",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the training config",
)
@click.option(
    "--model-name",
    type=str,
    help="Optional name to give to the trained model.",
)
@click.option(
    "--disable-logging",
    is_flag=True,
    help="Disables MLFlow logging.",
)
def train(model_config: Path, training_config: Path, model_name: str, disable_logging: bool):
    """
    Train a small GPT model on a Woolf corpus.
    """
    model_config_path = model_config
    model_config = load_yaml(model_config)
    training_config = load_yaml(training_config)
    training_config.name = model_name or getattr(training_config, "name", "woolfgpt")

    # Configuring MLflow, logging, and model versioning
    configure_mlflow("gpt", set_null=disable_logging)
    model_name, training_config = make_versioned_model_name(
        training_config, set_null=disable_logging
    )

    val_split = getattr(training_config.hparams, "val_split", 0.0)
    dataset = CorpusDataset(
        dataset_path=training_config.dataset,
        batch_size=training_config.hparams.batch_size,
        collate_for="torch",
        val_split=val_split,
    )
    train_loader = dataset.create_dataloader()
    val_loader = dataset.create_val_dataloader() if val_split > 0.0 else None

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Training with device: {device}")

    model = GPTTorch(GPTConfig(**model_config.model_params.to_dict())).to(device)

    # weight_decay=0.1; follows GPT-2 convention
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.hparams.lr, weight_decay=0.1
    )

    if training_config.lr_scheduler is not None:
        lr_scheduler = LR_SCHEDULER_REGISTRY[training_config.lr_scheduler.name](
            optimizer, **training_config.lr_scheduler.args.to_dict()
        )
    else:
        lr_scheduler = None

    lr_scheduling = {"scheduler": lr_scheduler, "warmup": training_config.lr_warmup}

    tokenizer = ByteLevelBPETokenizer(
        training_config.tokenizer + "/vocab.json",
        training_config.tokenizer + "/merges.txt",
    )

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("name", model_name)
        if not disable_logging:
            mlflow.log_text(Path(model_config_path).read_text(), "gpt_config.yml")
            mlflow.log_artifacts(str(training_config.tokenizer), artifact_path="tokenizer")
        gpt_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduling=lr_scheduling,
            device=device,
            tokenizer=tokenizer,
            num_epochs=training_config.hparams.epochs,
            early_stopping_patience=getattr(training_config.hparams, "early_stopping_patience", 0),
            early_stopping_min_delta=getattr(
                training_config.hparams, "early_stopping_min_delta", 0.0
            ),
            disable_logging=disable_logging,
        )


def create_train_step(model, optimizer):
    @jax.jit
    def train_step(params, opt_state, x, y, rng):
        def loss_fn(params):
            logits = model.apply({"params": params}, x, rngs={"dropout": rng}, training=True)
            loss = cross_entropy_loss(logits, y)
            return loss

        grads = jax.grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss = loss_fn(params)
        return params, opt_state, loss

    return train_step


@click.command()
@click.option(
    "--model-config",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the GPT model config yaml",
)
@click.option(
    "--training-config",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the training config",
)
@click.option(
    "--model-name",
    type=str,
    help="Optional name to give to the trained model.",
)
@click.option(
    "--disable-logging",
    is_flag=True,
    help="Disables MLFlow logging.",
)
def train_jax(model_config: Path, training_config: Path, model_name: str, disable_logging: bool):
    """
    Train a small GPT model on a Woolf corpus, using Flax Linen JAX model. The Flax Linen model
    and training was developed originally, but eventually dropped in favour of the NNX API.
    """

    # Load the configs
    model_config = load_yaml(model_config)
    training_config = load_yaml(training_config)

    dataset = CorpusDataset(
        dataset_path=training_config.dataset, batch_size=training_config.hparams.batch_size
    )
    train_loader = dataset.create_dataloader()

    key = jax.random.PRNGKey(0)

    # Instantiate the model
    model = GPTJax(GPTConfig(**model_config.model_params.to_dict()))
    dummy_input = jnp.ones(
        (training_config.hparams.batch_size, model.config.block_size), dtype=jnp.int32
    )
    params = model.init(key, dummy_input, training=False)["params"]

    # Get lr scheduler
    if training_config.lr_scheduler is not None:
        learning_rate_fn = LR_SCHEDULER_REGISTRY_JAX[training_config.lr_scheduler.name](
            **training_config.lr_scheduler.args.to_dict()
        )
    else:
        learning_rate_fn = lambda _: training_config.hparams.lr  # noqa: E731

    optimizer = optax.adamw(learning_rate_fn)
    opt_state = optimizer.init(params)

    train_step = create_train_step(model, optimizer)

    num_epochs = training_config.hparams.epochs

    with mlflow.start_run(run_name=model_name) if not disable_logging else nullcontext() as run:
        gpt_training_loop_jax(
            params,
            opt_state,
            learning_rate_fn,
            key,
            train_step,
            train_loader,
            num_epochs,
            disable_logging,
        )


@click.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the combined model + training config yaml",
)
@click.option(
    "--disable-logging",
    is_flag=True,
    help="Disables MLFlow logging.",
)
def train_jax_nnx(config: Path, disable_logging: bool):
    """
    Train a small GPT model on a Woolf corpus, using Flax NNX.
    """
    config = load_yaml(config)

    configure_mlflow("gpt", set_null=disable_logging)
    model_name, config = make_versioned_model_name(config, set_null=disable_logging)

    val_split = getattr(config.hparams, "val_split", 0.0)
    dataset = CorpusDataset(
        dataset_path=config.dataset,
        batch_size=config.hparams.batch_size,
        val_split=val_split,
    )
    train_loader = dataset.create_dataloader()
    val_loader = dataset.create_val_dataloader() if val_split > 0.0 else None

    tokenizer = ByteLevelBPETokenizer(
        config.tokenizer + "/vocab.json",
        config.tokenizer + "/merges.txt",
    )

    model = GPTNNX(**config.model_params.to_dict(), rngs=nnx.Rngs(0))

    if config.lr_scheduler is not None:
        learning_rate_fn = LR_SCHEDULER_REGISTRY_JAX[config.lr_scheduler.name](
            **config.lr_scheduler.args.to_dict()
        )
    else:
        learning_rate_fn = lambda _: config.hparams.lr  # noqa: E731

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate_fn, weight_decay=0.1),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    def loss_fn(model, x, y):
        logits = model(x, training=True)
        loss = cross_entropy_loss(logits, y)
        return loss, logits

    @nnx.jit
    def train_step(model, optimizer, metrics, x, y):
        """Single training step using Flax NNX."""
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, x, y)
        optimizer.update(model, grads)
        metrics.update(loss=loss, logits=logits, labels=y)
        return loss

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("name", model_name)
        gpt_training_loop_nnx(
            config,
            model,
            optimizer,
            metrics,
            train_loader,
            val_loader,
            learning_rate_fn,
            train_step,
            config.hparams.epochs,
            model_name=model_name,
            tokenizer=tokenizer,
            track_loss="train_loss",
            disable_logging=disable_logging,
        )


@click.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the fine-tuning config yaml",
)
@click.option("--disable-logging", is_flag=True, help="Disables MLFlow logging.")
def finetune(config: Path, disable_logging: bool):
    """
    Fine-tune GPT-2 small from HuggingFace on the Woolf corpus.
    Downloads the base model on first run (then cached). Saves the best checkpoint
    in HuggingFace format under gpt/artifacts/gpt2_finetuned/<model_name>/.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_cosine_schedule_with_warmup

    from woolfnet.data.dataset import GPT2CorpusDataset

    config = load_yaml(config)
    configure_mlflow("gpt", set_null=disable_logging)
    model_name, config = make_versioned_model_name(config, set_null=disable_logging)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Fine-tuning with device: {device}")

    hf_tokenizer = GPT2Tokenizer.from_pretrained(config.hf_model_name)
    hf_tokenizer.pad_token = hf_tokenizer.eos_token

    val_split = getattr(config.hparams, "val_split", 0.1)
    dataset = GPT2CorpusDataset(
        corpus_path=Path(config.corpus),
        tokenizer=hf_tokenizer,
        block_size=config.block_size,
        batch_size=config.hparams.batch_size,
        val_split=val_split,
    )
    train_loader = dataset.create_dataloader()
    val_loader = dataset.create_val_dataloader() if val_split > 0.0 else None

    model = GPT2LMHeadModel.from_pretrained(config.hf_model_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.hparams.lr, weight_decay=0.01)
    total_steps = len(train_loader) * config.hparams.epochs
    warmup_steps = getattr(config, "lr_warmup_steps", 100)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    save_dir = GPT_ARTIFACTS / "gpt2_finetuned" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(save_dir)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("name", model_name)
        mlflow.log_param("hf_model", config.hf_model_name)
        gpt_finetune_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            num_epochs=config.hparams.epochs,
            early_stopping_patience=getattr(config.hparams, "early_stopping_patience", 0),
            model_save_dir=save_dir,
            disable_logging=disable_logging,
        )


@click.command()
@click.option(
    "--our-model-config",
    type=click.Path(path_type=Path),
    default="gpt/configs/gpt_base.yml",
    show_default=True,
)
@click.option(
    "--our-model-weights",
    type=click.Path(path_type=Path),
    default="gpt/artifacts/model_weights/model.pt",
    show_default=True,
)
@click.option(
    "--our-tokenizer",
    type=str,
    default="data/tokenizers/woolf_both_corpus",
    show_default=True,
)
@click.option(
    "--finetuned-model",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to fine-tuned GPT-2 directory (HuggingFace format)",
)
@click.option(
    "--corpus",
    type=click.Path(path_type=Path),
    default="data/corpora/woolf_both_corpus.txt",
    show_default=True,
)
@click.option("--val-split", type=float, default=0.1, show_default=True)
def evaluate(
    our_model_config: Path,
    our_model_weights: Path,
    our_tokenizer: str,
    finetuned_model: Path,
    corpus: Path,
    val_split: float,
):
    """
    Compare our from-scratch GPT vs fine-tuned GPT-2 on NLP metrics:
    perplexity, bits-per-character (BPC), and Distinct-1/2 lexical diversity.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    from woolfnet.gpt.evaluation import EVAL_PROMPTS, run_comparison

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model_cfg = load_yaml(our_model_config)
    our_model = GPTTorch(GPTConfig(**model_cfg.model_params.to_dict())).to(device)
    our_model.load_state_dict(torch.load(our_model_weights, map_location=device, weights_only=True))

    our_tok = ByteLevelBPETokenizer(our_tokenizer + "/vocab.json", our_tokenizer + "/merges.txt")

    hf_model = GPT2LMHeadModel.from_pretrained(str(finetuned_model), local_files_only=True).to(
        device
    )
    hf_tokenizer = GPT2Tokenizer.from_pretrained(str(finetuned_model), local_files_only=True)

    full_text = Path(corpus).read_text(encoding="utf-8")
    val_text = full_text[int(len(full_text) * (1 - val_split)) :]

    results = run_comparison(
        our_model,
        our_tok,
        model_cfg.model_params.block_size,
        hf_model,
        hf_tokenizer,
        val_text,
        device,
    )

    print("\n=== Model Comparison ===\n")
    print(f"{'Metric':<20} {'Our GPT':>10} {'Fine-tuned GPT-2':>18}")
    print("-" * 50)
    print(
        f"{'Perplexity':<20} {results['perplexity']['our']:>10.2f} {results['perplexity']['gpt2']:>18.2f}"
    )
    print(f"{'BPC':<20} {results['bpc']['our']:>10.4f} {results['bpc']['gpt2']:>18.4f}")
    print(
        f"{'Distinct-1':<20} {results['distinct_1']['our']:>10.4f} {results['distinct_1']['gpt2']:>18.4f}"
    )
    print(
        f"{'Distinct-2':<20} {results['distinct_2']['our']:>10.4f} {results['distinct_2']['gpt2']:>18.4f}"
    )

    print()
    for prompt, our_sample, hf_sample in zip(
        EVAL_PROMPTS, results["samples"]["our"], results["samples"]["gpt2"]
    ):
        print(f"Prompt: {prompt!r}")
        print(f"  Our GPT: {our_sample}")
        print(f"  GPT-2:   {hf_sample}")
        print()


COMMANDS = {
    "train": train,
    "train_jax": train_jax,
    "train_jax_nnx": train_jax_nnx,
    "finetune": finetune,
    "evaluate": evaluate,
}
