"""SDK-22 HuggingFace transformers hook — unit tests.

Skipped in environments without ``transformers`` (or ``torch``) so the
core CI path stays green. When available, we exercise the
``TrainerCallback`` auto-attach on ``Trainer.__init__``, the
``epoch`` / ``step`` scope shape, ``loss`` / ``learning_rate`` marks,
nesting under torch hooks (SDK-20), subclass safety, and clean
``uninstall``.
"""

from __future__ import annotations

import logging

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
from cirron.hooks._torch_impl import install as torch_install  # noqa: E402
from cirron.hooks._transformers_impl import install as tr_install  # noqa: E402


class _LossModel(nn.Module):
    """Tiny model that returns a HF-style ``{"loss": ...}`` dict so
    ``Trainer.compute_loss`` can read ``outputs["loss"]`` (its dict
    branch) without us having to construct a full ``ModelOutput``."""

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


def _make_args(tmp_path, epochs: int = 2, batch_size: int = 2) -> TrainingArguments:
    return TrainingArguments(
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


@pytest.fixture(autouse=True)
def _reset_default_stack():
    stack = get_default_stack()
    stack.drain_closed_all()
    buf = get_default_mark_buffer()
    buf.drain_all()
    yield
    stack.drain_closed_all()
    buf.drain_all()


@pytest.fixture
def stack():
    return get_default_stack()


@pytest.fixture
def ci(tmp_path):
    return Cirron(output_dir=str(tmp_path))


def _make_trainer(tmp_path, *, trainer_cls=Trainer, epochs=2, batch_size=2, n=6):
    return trainer_cls(
        model=_LossModel(),
        args=_make_args(tmp_path, epochs=epochs, batch_size=batch_size),
        train_dataset=_TinyDataset(n=n),
    )


def test_install_returns_handle_with_transformers_name(stack, ci):
    h = tr_install(stack, ci)
    try:
        assert h.name == "transformers"
    finally:
        h.uninstall()


def test_uninstall_restores_init(stack, ci):
    orig_init = Trainer.__init__
    h = tr_install(stack, ci)
    assert Trainer.__init__ is not orig_init
    h.uninstall()
    assert Trainer.__init__ is orig_init


def test_double_uninstall_is_noop(stack, ci):
    h = tr_install(stack, ci)
    h.uninstall()
    h.uninstall()


def test_train_produces_epoch_and_step_scope_tree(stack, ci, tmp_path):
    h = tr_install(stack, ci)
    try:
        trainer = _make_trainer(tmp_path, epochs=2, batch_size=2, n=6)
        trainer.train()
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    epochs = [s for s in closed if s.name == "epoch"]
    steps = [s for s in closed if s.name == "step"]
    # Two epochs (state.epoch is 0.0 then 1.0 at on_epoch_begin).
    assert [s.index for s in epochs] == [0, 1]
    # 6 samples / 2 = 3 steps per epoch * 2 epochs = 6 steps.
    assert len(steps) == 6


def test_steps_are_children_of_their_epoch(stack, ci, tmp_path):
    h = tr_install(stack, ci)
    try:
        trainer = _make_trainer(tmp_path, epochs=1, batch_size=2, n=4)
        trainer.train()
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    epoch = next(s for s in closed if s.name == "epoch")
    steps = [s for s in closed if s.name == "step"]
    assert steps  # sanity
    for st in steps:
        assert st.parent_id == epoch.id


def test_torch_hooks_nest_inside_transformers_step(stack, ci, tmp_path):
    """AC #2: torch forward/backward/optimizer_step nest inside the step span."""
    th = torch_install(stack, ci)
    h = tr_install(stack, ci)
    try:
        trainer = _make_trainer(tmp_path, epochs=1, batch_size=2, n=4)
        trainer.train()
    finally:
        h.uninstall()
        th.uninstall()
    closed = stack.drain_closed_all()
    by_id = {s.id: s for s in closed}
    step_ids = {s.id for s in closed if s.name == "step"}

    def _has_step_ancestor(s) -> bool:
        cur = s
        guard = 64
        while cur is not None and guard > 0:
            guard -= 1
            if cur.id in step_ids:
                return True
            if cur.parent_id is None:
                return False
            cur = by_id.get(cur.parent_id)
        return False

    torch_spans = [s for s in closed if s.name in {"forward", "backward", "optimizer_step"}]
    assert torch_spans, "expected at least one torch hook span"
    for sp in torch_spans:
        # Skip the outermost forward span if the span itself happens to
        # be a step ancestor (e.g. eval forwards before train begins) —
        # we only care that *every* torch span produced inside a training
        # step has the step in its ancestor chain.
        assert _has_step_ancestor(sp), (
            f"{sp.name} span (id={sp.id}, parent={sp.parent_id}) is not nested inside a step scope"
        )


def test_callback_attaches_on_subclassed_trainer(stack, ci, tmp_path):
    """AC #3: subclassed Trainer must still get the callback attached
    and its own ``__init__`` body must run."""

    class MyTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.extra = 1

    h = tr_install(stack, ci)
    try:
        trainer = _make_trainer(tmp_path, trainer_cls=MyTrainer, epochs=1, batch_size=2, n=4)
        assert trainer.extra == 1
        trainer.train()
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    epochs = [s for s in closed if s.name == "epoch"]
    steps = [s for s in closed if s.name == "step"]
    assert epochs and steps


def test_callback_not_duplicated_across_trainer_instances(stack, ci, tmp_path):
    h = tr_install(stack, ci)
    try:
        callback_cls = h.CirronTrainerCallback  # type: ignore[attr-defined]
        t1 = _make_trainer(tmp_path / "a", epochs=1, batch_size=2, n=4)
        t2 = _make_trainer(tmp_path / "b", epochs=1, batch_size=2, n=4)
        for tr in (t1, t2):
            count = sum(1 for cb in tr.callback_handler.callbacks if isinstance(cb, callback_cls))
            assert count == 1
    finally:
        h.uninstall()


def test_loss_and_lr_marks_captured(stack, ci, tmp_path):
    h = tr_install(stack, ci)
    try:
        trainer = _make_trainer(tmp_path, epochs=1, batch_size=2, n=4)
        trainer.train()
    finally:
        h.uninstall()
    marks = get_default_mark_buffer().drain_all()
    names = {m.name for m in marks}
    assert "loss" in names, f"loss not in marks: {names}"
    assert "learning_rate" in names, f"learning_rate not in marks: {names}"
    lrs = [m for m in marks if m.name == "learning_rate"]
    assert all(isinstance(m.value, float) for m in lrs)


def test_handler_exception_does_not_crash_training(stack, ci, tmp_path, monkeypatch, caplog):
    from cirron.core.scope import ScopeStack

    h = tr_install(stack, ci)
    try:
        monkeypatch.setattr(
            ScopeStack,
            "push",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with caplog.at_level(logging.WARNING, logger="cirron.hooks.transformers"):
            trainer = _make_trainer(tmp_path, epochs=1, batch_size=2, n=4)
            # train() must complete despite scope.push raising on every call.
            trainer.train()
    finally:
        h.uninstall()
