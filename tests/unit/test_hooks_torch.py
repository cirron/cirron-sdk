"""SDK-20 PyTorch hooks — unit tests.

Skipped in environments without ``torch`` so the core CI path (no
frameworks installed) stays green. When ``torch`` is available, we
exercise each hook target independently and confirm ``uninstall()``
restores the originals.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cirron.core.config import Cirron  # noqa: E402
from cirron.core.scope import ScopeStack  # noqa: E402
from cirron.hooks._torch_impl import install as torch_install  # noqa: E402


def _names(scopes):
    return [s.name for s in scopes]


@pytest.fixture
def stack():
    return ScopeStack()


@pytest.fixture
def ci(tmp_path):
    return Cirron(output_dir=str(tmp_path))


def test_install_returns_handle_with_torch_name(stack, ci):
    h = torch_install(stack, ci)
    try:
        assert h.name == "torch"
    finally:
        h.uninstall()


def test_uninstall_restores_originals(stack, ci):
    # Optimizer.step uses PyTorch's global step hooks (not monkey-patched),
    # so identity is unchanged; the other three are patched.
    orig_tensor_bw = torch.Tensor.backward
    orig_autograd_bw = torch.autograd.backward
    orig_dl_iter = torch.utils.data.DataLoader.__iter__

    h = torch_install(stack, ci)
    assert torch.Tensor.backward is not orig_tensor_bw
    assert torch.autograd.backward is not orig_autograd_bw
    assert torch.utils.data.DataLoader.__iter__ is not orig_dl_iter

    h.uninstall()

    assert torch.Tensor.backward is orig_tensor_bw
    assert torch.autograd.backward is orig_autograd_bw
    assert torch.utils.data.DataLoader.__iter__ is orig_dl_iter


def test_double_uninstall_is_noop(stack, ci):
    h = torch_install(stack, ci)
    h.uninstall()
    # Should not raise or re-restore anything.
    h.uninstall()


def test_forward_hook_fires_on_module_call(stack, ci):
    h = torch_install(stack, ci)
    try:
        model = torch.nn.Linear(4, 2)
        model(torch.zeros(1, 4))
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    assert "forward" in _names(closed)


def test_forward_only_top_level_scope(stack, ci):
    """Nested submodules must not each produce their own forward span."""
    h = torch_install(stack, ci)
    try:
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )
        model(torch.zeros(1, 4))
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    assert _names(closed).count("forward") == 1


def test_backward_and_optimizer_step_produce_scopes(stack, ci):
    h = torch_install(stack, ci)
    try:
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.zeros(1, 4)
        y = torch.zeros(1, 2)
        out = model(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        opt.step()
    finally:
        h.uninstall()
    names = _names(stack.drain_closed_all())
    assert "forward" in names
    assert "backward" in names
    assert "optimizer_step" in names


def test_dataloader_data_load_scope(stack, ci):
    h = torch_install(stack, ci)
    try:
        xs = torch.zeros(6, 4)
        ys = torch.zeros(6, 2)
        ds = torch.utils.data.TensorDataset(xs, ys)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        for _ in loader:
            pass
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    names = _names(closed)
    assert "data_load" in names
    # data_load spans should carry a stall attribute.
    dl_spans = [s for s in closed if s.name == "data_load"]
    assert any("data_load_ns" in s.attrs for s in dl_spans)


def test_two_epochs_produce_two_epoch_scopes(stack, ci):
    h = torch_install(stack, ci)
    try:
        xs = torch.zeros(4, 4)
        ys = torch.zeros(4, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=2)
        for _ in range(2):
            for _ in loader:
                pass
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    epochs = [s for s in closed if s.name == "epoch"]
    assert len(epochs) == 2
    assert [s.index for s in epochs] == [0, 1]


def test_inference_only_model_no_optimizer(stack, ci):
    """Model used only for forward inference shouldn't crash or miss spans."""
    h = torch_install(stack, ci)
    try:
        model = torch.nn.Linear(4, 2)
        with torch.no_grad():
            model(torch.zeros(1, 4))
            model(torch.zeros(1, 4))
    finally:
        h.uninstall()
    names = _names(stack.drain_closed_all())
    assert names.count("forward") == 2
    assert "backward" not in names
    assert "optimizer_step" not in names


def test_custom_module_subclass_is_traced(stack, ci):
    class MyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    h = torch_install(stack, ci)
    try:
        MyNet()(torch.zeros(1, 4))
    finally:
        h.uninstall()
    assert "forward" in _names(stack.drain_closed_all())


def test_hook_exception_is_caught(stack, ci, monkeypatch, caplog):
    """A scope-push exception must not crash user code."""
    h = torch_install(stack, ci)
    try:
        # Poison ScopeStack.push so any hook that tries to open a scope
        # raises. The _catch wrapper in the torch impl should swallow it.
        monkeypatch.setattr(
            ScopeStack, "push", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        with caplog.at_level("WARNING", logger="cirron.hooks.torch"):
            model = torch.nn.Linear(4, 2)
            # Must not raise.
            out = model(torch.zeros(1, 4))
        assert out.shape == (1, 2)
    finally:
        h.uninstall()


def test_no_torch_usage_after_uninstall_produces_no_spans(stack, ci):
    h = torch_install(stack, ci)
    h.uninstall()
    # With hooks gone, nothing should land in the stack.
    model = torch.nn.Linear(4, 2)
    model(torch.zeros(1, 4))
    closed = stack.drain_closed_all()
    assert closed == []


def test_many_epochs_do_not_blow_stack_depth(stack, ci):
    """Regression for PR#20 comments #1/#2: rotating epochs must actually
    leave the stack, not just be close_scope'd in place. A long run should
    stay well below ``MAX_DEPTH`` after uninstall."""
    h = torch_install(stack, ci)
    try:
        xs = torch.zeros(2, 4)
        ys = torch.zeros(2, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=1)
        for _ in range(50):
            for _ in loader:
                pass
    finally:
        h.uninstall()
    # After uninstall, no scopes should be left open on the current thread.
    assert stack.current() is None
    # All 50 epoch spans should be drainable.
    closed = stack.drain_closed_all()
    epochs = [s for s in closed if s.name == "epoch"]
    assert len(epochs) == 50
    assert [s.index for s in epochs] == list(range(50))


def test_epoch_scopes_are_siblings_not_nested(stack, ci):
    """PR#20 #1: consecutive epochs must not be parent-child of each other."""
    h = torch_install(stack, ci)
    try:
        xs = torch.zeros(2, 4)
        ys = torch.zeros(2, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=2)
        for _ in range(3):
            for _ in loader:
                pass
    finally:
        h.uninstall()
    epochs = [s for s in stack.drain_closed_all() if s.name == "epoch"]
    assert len(epochs) == 3
    # No epoch should claim another epoch as its parent.
    epoch_ids = {s.id for s in epochs}
    for s in epochs:
        assert s.parent_id not in epoch_ids, f"epoch {s.index} is nested under another epoch"


def test_stopiteration_does_not_emit_data_load_span(stack, ci):
    """PR#20 #3: exhausting the iterator must not produce a trailing span."""
    h = torch_install(stack, ci)
    try:
        xs = torch.zeros(4, 4)
        ys = torch.zeros(4, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=2)
        # 4 samples / batch 2 = exactly 2 batches → 2 data_load spans, not 3.
        for _ in loader:
            pass
    finally:
        h.uninstall()
    data_loads = [s for s in stack.drain_closed_all() if s.name == "data_load"]
    assert len(data_loads) == 2


def test_close_preserves_unrelated_scope_on_top(stack, ci):
    """PR#20 #4: if a user scope was opened on top of our forward span,
    _close must fall back to close_scope rather than popping the user's
    scope. The user scope must survive until they close it themselves."""
    from cirron.core.scope import get_current_scope  # local import: uses default stack

    h = torch_install(stack, ci)
    try:
        # Mirror the hook's manual sequence by pushing a scope via our
        # local stack, then simulating a "user opened something on top"
        # before _close runs. We use the real forward hook entry point
        # to exercise the same _close path the framework uses.
        model = torch.nn.Linear(4, 2)

        # Monkey-patch the post-hook path indirectly by opening a scope
        # inside forward. nn.Module forward hooks fire around __call__,
        # so opening a scope from within Linear.forward isn't possible
        # without subclassing — subclass it.
        class Wrapped(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.fc(x)
                # User opens a scope before the outer forward post-hook
                # fires. The post-hook's _close should not pop this.
                stack.push("user", foo="bar")
                return out

        Wrapped()(torch.zeros(1, 4))
        # The user scope should still be on the stack; the forward scope
        # should already be closed via close_scope.
        cur = stack.current()
        assert cur is not None and cur.name == "user", (
            f"user scope was popped by _close; got {cur!r}"
        )
        # Clean up the user scope so subsequent asserts see a clean stack.
        stack.pop()
    finally:
        h.uninstall()
    names = [s.name for s in stack.drain_closed_all()]
    # Both the forward span (closed via close_scope) and the user span
    # (closed via pop) should have landed.
    assert "forward" in names
    assert "user" in names
    del get_current_scope  # silence unused-import warning


def test_epoch_step_threshold_fallback(stack, tmp_path):
    """Without a DataLoader, optimizer steps past the threshold rotate the epoch."""
    ci = Cirron(output_dir=str(tmp_path))
    ci._profile_config = {"torch": {"epoch_steps": 2}}
    h = torch_install(stack, ci)
    try:
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(3):
            out = model(torch.zeros(1, 2))
            loss = out.sum()
            loss.backward()
            opt.step()
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    epochs = [s for s in closed if s.name == "epoch"]
    # Two rotations on the optimizer path (after step 2 and 4 that never
    # fires — we only did 3 steps — so at least one epoch rotation).
    assert len(epochs) >= 1
