"""Integration test: the real ``install_hooks`` dispatcher with both real hooks.

Scope note on what this adds over the unit tests. ``test_hooks_transformers.py``
already co-installs the two REAL installers (``tr_install`` then
``torch_install``) through a hand-built ``HookContext`` and asserts single
epoch ownership. What it does NOT cover, and what this file does, is the
production entry point itself: ``install_hooks()``, which

  1. sorts the requested frameworks by ``_FRAMEWORK_PRIORITY`` (transformers
     before torch) regardless of the order the caller passed, and
  2. constructs the single shared ``HookContext`` that lets transformers claim
     ``"epoch"`` before torch decides whether to open its own.

Both of those are covered today only by STUB installers
(``test_hook_registry.py``). Joining the real dispatcher to the real installers
and a real ``Trainer`` run is the gap this closes, and it doubles as the
canary for transformers 5.x compatibility.

The tiny model / dataset / args harness is copied from
``tests/unit/test_hooks_transformers.py`` rather than imported, so this
integration test does not depend on unit-test internals.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")  # transformers Trainer (PyTorch) requires it

import torch.nn as nn  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import Trainer, TrainingArguments  # noqa: E402

from cirron.core.config import Cirron  # noqa: E402
from cirron.core.mark import get_default_mark_buffer  # noqa: E402
from cirron.core.scope import get_default_stack  # noqa: E402
from cirron.hooks._registry import install_hooks  # noqa: E402


class _LossModel(nn.Module):
    """Returns a HF-style dict so ``Trainer.compute_loss`` takes its dict branch."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.linear(x)
        loss = ((out - labels) ** 2).mean()
        return {"loss": loss, "logits": out}


class _TinyDataset(Dataset):
    def __init__(self, n: int = 6) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.zeros(4, dtype=torch.float32),
            "labels": torch.zeros(2, dtype=torch.float32),
        }


def _make_trainer(tmp_path, *, epochs=2, batch_size=2, n=6):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=1,
        report_to=[],
        disable_tqdm=True,
        save_strategy="no",
        eval_strategy="no",
        use_cpu=True,
        dataloader_num_workers=0,
        log_level="error",
    )
    return Trainer(model=_LossModel(), args=args, train_dataset=_TinyDataset(n=n))


@pytest.fixture(autouse=True)
def _reset_default_state():
    stack = get_default_stack()
    stack.drain_closed_all()
    buf = get_default_mark_buffer()
    buf.drain_all()
    yield
    stack.drain_closed_all()
    buf.drain_all()


def test_install_hooks_yields_single_epoch_ownership(tmp_path):
    """``install_hooks`` must reorder to transformers-before-torch and share one
    context, so a real Trainer run produces exactly one ``epoch`` span per epoch."""
    stack = get_default_stack()
    cirron = Cirron(output_dir=str(tmp_path))

    # Deliberately WRONG order: if the priority sort did not run, torch would
    # install first, fail to see the "epoch" claim, and open its own rotation
    # epochs on top of the transformers ones.
    handles = install_hooks(["torch", "transformers"], stack, cirron)
    try:
        assert [h.name for h in handles] == ["transformers", "torch"], (
            f"priority sort did not run; got {[h.name for h in handles]}"
        )
        trainer = _make_trainer(tmp_path, epochs=2, batch_size=2, n=6)
        trainer.train()
    finally:
        for h in reversed(handles):
            h.uninstall()

    closed = stack.drain_closed_all()

    epochs = [s for s in closed if s.name == "epoch"]
    assert len(epochs) == 2, (
        f"expected 2 epoch spans (one per real epoch), got {len(epochs)}: "
        f"{[(s.index, s.parent_id) for s in epochs]}"
    )
    assert [s.index for s in epochs] == [0, 1]

    # Step spans are singly owned: every step hangs off an epoch span.
    epoch_ids = {s.id for s in epochs}
    steps = [s for s in closed if s.name == "step"]
    assert steps, "expected at least one step span"
    for st in steps:
        assert st.parent_id in epoch_ids, (
            f"step span (id={st.id}) parent {st.parent_id} is not one of the epoch spans"
        )
