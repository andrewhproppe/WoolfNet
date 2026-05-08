from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import mlflow
import optax
import orbax.checkpoint as ocp
import torch
from flax import nnx
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.nn import CrossEntropyLoss

from woolfnet.config import DotDict
from woolfnet.gpt import GPT_ARTIFACTS
from woolfnet.gpt.metrics import cross_entropy_loss
from woolfnet.gpt.utils import generate_text, generate_text_nnx, logger, warmup_optimizer


def create_train_pbar():
    """
    Creates a general model training progress bar that displays the error and lr at each
    step.
    """
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        TextColumn("• Error: {task.fields[error]:.4f}"),
        TextColumn("• lr: {task.fields[lr]:.5f}"),
    )


def gpt_training_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduling,
    device,
    tokenizer,
    num_epochs,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    disable_logging: bool = False,
):
    lr_scheduler = lr_scheduling["scheduler"]
    base_lr = optimizer.param_groups[0]["lr"]
    warmup_epochs = lr_scheduling["warmup"].warmup_epochs

    progress = create_train_pbar()

    best_val_loss = float("inf")
    no_improve_count = 0

    weights_dir = Path(GPT_ARTIFACTS / "model_weights")
    weights_dir.mkdir(exist_ok=True, parents=True)

    with Live(progress, refresh_per_second=5):
        for epoch in range(num_epochs):
            task = progress.add_task(
                f"Step progression for epoch {epoch}",
                total=len(train_loader),
                error=0.0,
                lr=optimizer.param_groups[0]["lr"],
            )
            model.train()
            total_loss = 0

            if epoch < warmup_epochs:
                optimizer = warmup_optimizer(
                    optimizer=optimizer,
                    epoch=epoch,
                    warmup_epochs=warmup_epochs,
                    base_lr=base_lr,
                    warmup_start_lr=lr_scheduling["warmup"].warmup_start_lr,
                )
                progress.update(task, advance=0, lr=optimizer.param_groups[0]["lr"])

            for step, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                y_pred = model(x)
                loss = CrossEntropyLoss()(y_pred.view(-1, model.config.vocab_size), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                progress.update(task, advance=1, error=total_loss / (step + 1))

            progress.remove_task(task)
            avg_train_loss = total_loss / len(train_loader)

            # Validation pass
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    val_loss_accum += CrossEntropyLoss()(
                        y_pred.view(-1, model.config.vocab_size), y.view(-1)
                    ).item()
            avg_val_loss = val_loss_accum / len(val_loader)
            model.train()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}: Train {avg_train_loss:.4f}  Val {avg_val_loss:.4f}  lr {current_lr:.6f}"
            )

            if not disable_logging:
                mlflow.log_metrics(
                    {"train_loss": avg_train_loss, "val_loss": avg_val_loss, "lr": current_lr},
                    step=epoch,
                )

            if avg_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = avg_val_loss
                no_improve_count = 0
                torch.save(model.state_dict(), weights_dir / "model.pt")
                logger.info(f"Val loss improved to {avg_val_loss:.4f} — model saved")
            else:
                no_improve_count += 1

            if lr_scheduler is not None and epoch >= warmup_epochs:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(avg_val_loss)
                else:
                    lr_scheduler.step()

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train: {avg_train_loss:.4f}  Val: {avg_val_loss:.4f}",
                flush=True,
            )

            if early_stopping_patience > 0 and no_improve_count >= early_stopping_patience:
                logger.info(
                    f"Early stopping: val_loss has not improved for {early_stopping_patience} epochs."
                )
                print(
                    f"Early stopping at epoch {epoch + 1} (no val improvement for {early_stopping_patience} epochs)."
                )
                break

            prompt = "Mrs. Dalloway said "
            sample_text = generate_text(model, tokenizer, prompt, max_new_tokens=50, device=device)
            logger.info(sample_text)


def gpt_training_loop_jax(
    params: Any,
    opt_state: optax.OptState,
    learning_rate_fn: Callable[[int], float],
    key: jax.Array,
    train_step: Callable,
    train_loader: Iterator[Tuple[jax.Array, jax.Array]],
    num_epochs: int,
    disable_logging: bool,
):
    """
    Training loop for JAX GPT2 model using Flax Linen.
    """
    progress = create_train_pbar()

    global_step = 0
    best_loss = float("inf")
    loss = float("inf")
    with Live(progress, refresh_per_second=5):
        for epoch in range(num_epochs):
            task = progress.add_task(
                f"Step progression for epoch {epoch}",
                total=len(train_loader),
                error=0.0,
                lr=float(learning_rate_fn(global_step)),
            )

            for x_batch, y_batch in train_loader:
                key, subkey = jax.random.split(key)
                x_batch = jnp.array(x_batch)
                y_batch = jnp.array(y_batch)
                params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch, subkey)

                lr = float(learning_rate_fn(global_step))

                progress.update(task, advance=1, error=loss, lr=lr)

                mlflow.log_metrics({"train_loss": float(loss), "lr": lr}, step=global_step)
                global_step += 1

            if loss < best_loss:
                best_loss = loss

            progress.remove_task(task)

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")


@nnx.jit
def _nnx_eval_step(model, x: jax.Array, y: jax.Array) -> jax.Array:
    """Single eval step — no gradient, dropout disabled via training=False."""
    return cross_entropy_loss(model(x, training=False), y)


def gpt_training_loop_nnx(
    config: DotDict,
    model,
    optimizer,
    metrics,
    train_loader: Iterator[Tuple[jax.Array, jax.Array]],
    val_loader: Optional[Iterator[Tuple[jax.Array, jax.Array]]],
    learning_rate_fn: Callable[[int], float],
    train_step: Callable,
    num_epochs: int,
    model_name: str,
    tokenizer=None,
    track_loss: str = "train_loss",
    eval_every_n_steps: int = 5,
    generate_every_n_epochs: int = 10,
    disable_logging: bool = False,
):
    """
    Training loop for JAX GPT2 model using Flax NNX.
    """
    progress = create_train_pbar()

    metrics_history = {"train_loss": [], "step": []}

    global_step = 0
    best_val_loss = float("inf")
    loss = 0.0
    lr = 0.0

    if not disable_logging:
        ckpt_dir = GPT_ARTIFACTS / "nnx-checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_manager = ocp.CheckpointManager(
            ckpt_dir / model_name,
            options=ocp.CheckpointManagerOptions(max_to_keep=2),
        )

    with Live(progress, refresh_per_second=5):
        for epoch in range(num_epochs):
            task = progress.add_task(
                f"Step progression for epoch {epoch}",
                total=len(train_loader),
                error=0.0,
                lr=float(learning_rate_fn(global_step)),
            )

            for x_batch, y_batch in train_loader:
                train_step(model, optimizer, metrics, x_batch, y_batch)

                if global_step > 0 and (global_step % eval_every_n_steps == 0):
                    for metric, value in metrics.compute().items():
                        metrics_history[f"train_{metric}"].append(value)
                        metrics_history["step"].append(global_step)
                    loss = float(metrics_history[track_loss][-1])
                    metrics.reset()

                    lr = float(learning_rate_fn(global_step))

                    if not disable_logging:
                        mlflow.log_metrics({"train_loss": loss, "lr": lr}, step=global_step)

                global_step += 1
                progress.update(task, advance=1, error=loss, lr=lr)

            progress.remove_task(task)

            # Validation pass
            val_loss = None
            if val_loader is not None:
                val_losses = [float(_nnx_eval_step(model, x, y)) for x, y in val_loader]
                val_loss = sum(val_losses) / len(val_losses)
                if not disable_logging:
                    mlflow.log_metrics({"val_loss": val_loss}, step=global_step)

            val_loss_str = f"  Val: {val_loss:.4f}" if val_loss is not None else ""
            print(f"Epoch {epoch + 1}/{num_epochs}, Train: {loss:.4f}{val_loss_str}")

            # Checkpoint on val_loss improvement (falls back to train_loss if no val)
            tracked = val_loss if val_loss is not None else loss
            if tracked < best_val_loss:
                logger.info(f"Loss improved {best_val_loss:.4f} → {tracked:.4f}")
                best_val_loss = tracked
                if not disable_logging:
                    _, state = nnx.split(model)
                    checkpoint_manager.save(global_step, args=ocp.args.StandardSave(state))
                    checkpoint_manager.wait_until_finished()
                    logger.info(f"Checkpoint saved at step {global_step}")

            # Periodic text generation for qualitative assessment
            if tokenizer is not None and (epoch + 1) % generate_every_n_epochs == 0:
                sample = generate_text_nnx(
                    model, tokenizer, "Mrs. Dalloway said ", model.block_size, max_new_tokens=60
                )
                logger.info(f"[epoch {epoch + 1}] {sample}")
                print(f"  Sample: {sample}")

        if not disable_logging:
            mlflow.log_artifacts(ckpt_dir / model_name)


def gpt_finetune_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    device: str,
    num_epochs: int,
    early_stopping_patience: int = 0,
    model_save_dir: Optional[Path] = None,
    disable_logging: bool = False,
):
    """
    Fine-tuning loop for a HuggingFace GPT-2 model. The LR scheduler is stepped every optimizer step
    (not per epoch), as is standard for HF cosine-with-warmup schedules.
    """
    progress = create_train_pbar()
    best_val_loss = float("inf")
    no_improve_count = 0
    global_step = 0

    with Live(progress, refresh_per_second=5):
        for epoch in range(num_epochs):
            task = progress.add_task(
                f"Step progression for epoch {epoch}",
                total=len(train_loader),
                error=0.0,
                lr=optimizer.param_groups[0]["lr"],
            )
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = model(input_ids=batch, labels=batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                total_loss += loss.item()
                global_step += 1
                current_lr = optimizer.param_groups[0]["lr"]
                progress.update(task, advance=1, error=total_loss / (step + 1), lr=current_lr)

            progress.remove_task(task)
            avg_train_loss = total_loss / len(train_loader)

            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    val_loss_accum += model(input_ids=batch, labels=batch).loss.item()
            avg_val_loss = val_loss_accum / len(val_loader)
            model.train()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}: Train {avg_train_loss:.4f}  Val {avg_val_loss:.4f}  lr {current_lr:.6f}"
            )  # noqa: E501
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train: {avg_train_loss:.4f}  Val: {avg_val_loss:.4f}"
            )  # noqa: E501

            if not disable_logging:
                mlflow.log_metrics(
                    {"train_loss": avg_train_loss, "val_loss": avg_val_loss, "lr": current_lr},
                    step=epoch,
                )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_count = 0
                if model_save_dir is not None:
                    model.save_pretrained(model_save_dir)
                    logger.info(f"Val loss improved to {avg_val_loss:.4f} — model saved")
            else:
                no_improve_count += 1

            if early_stopping_patience > 0 and no_improve_count >= early_stopping_patience:
                logger.info(
                    f"Early stopping: val_loss has not improved for {early_stopping_patience} epochs."
                )  # noqa: E501
                print(f"Early stopping at epoch {epoch + 1}.")
                break
