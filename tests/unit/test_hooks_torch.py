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
from cirron.hooks._registry import HookContext  # noqa: E402
from cirron.hooks._torch_impl import install as torch_install  # noqa: E402


def _names(scopes):
    return [s.name for s in scopes]


@pytest.fixture
def stack():
    return ScopeStack()


@pytest.fixture
def ci(tmp_path):
    return Cirron(output_dir=str(tmp_path))


@pytest.fixture
def ctx():
    return HookContext()


def test_install_returns_handle_with_torch_name(stack, ci, ctx):
    h = torch_install(stack, ci, ctx)
    try:
        assert h.name == "torch"
    finally:
        h.uninstall()


def test_uninstall_restores_originals(stack, ci, ctx):
    # Optimizer.step uses PyTorch's global step hooks (not monkey-patched),
    # so identity is unchanged; the other three are patched.
    orig_tensor_bw = torch.Tensor.backward
    orig_autograd_bw = torch.autograd.backward
    orig_dl_iter = torch.utils.data.DataLoader.__iter__

    h = torch_install(stack, ci, ctx)
    assert torch.Tensor.backward is not orig_tensor_bw
    assert torch.autograd.backward is not orig_autograd_bw
    assert torch.utils.data.DataLoader.__iter__ is not orig_dl_iter

    h.uninstall()

    assert torch.Tensor.backward is orig_tensor_bw
    assert torch.autograd.backward is orig_autograd_bw
    assert torch.utils.data.DataLoader.__iter__ is orig_dl_iter


def test_double_uninstall_is_noop(stack, ci, ctx):
    h = torch_install(stack, ci, ctx)
    h.uninstall()
    # Should not raise or re-restore anything.
    h.uninstall()


def test_forward_hook_fires_on_module_call(stack, ci, ctx):
    h = torch_install(stack, ci, ctx)
    try:
        model = torch.nn.Linear(4, 2)
        model(torch.zeros(1, 4))
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    assert "forward" in _names(closed)


def test_forward_only_top_level_scope(stack, ci, ctx):
    """Nested submodules must not each produce their own forward span."""
    h = torch_install(stack, ci, ctx)
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


def test_backward_and_optimizer_step_produce_scopes(stack, ci, ctx):
    h = torch_install(stack, ci, ctx)
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


def test_dataloader_data_load_scope(stack, ci, ctx):
    h = torch_install(stack, ci, ctx)
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


def test_two_epochs_produce_two_epoch_scopes(stack, ci, ctx):
    h = torch_install(stack, ci, ctx)
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


def test_forward_scope_has_mode_attr(stack, ci, ctx):
    """Forward spans carry ``mode=train|eval`` so trace consumers can
    distinguish training forwards from inference forwards without
    reaching into the model."""
    h = torch_install(stack, ci, ctx)
    try:
        model = torch.nn.Linear(4, 2)
        model.train(True)
        model(torch.zeros(1, 4))
        model.train(False)
        model(torch.zeros(1, 4))
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    forwards = [s for s in closed if s.name == "forward"]
    assert len(forwards) == 2
    modes = [s.attrs.get("mode") for s in forwards]
    assert modes == ["train", "eval"]


def test_inference_only_model_no_optimizer(stack, ci, ctx):
    """Model used only for forward inference shouldn't crash or miss spans."""
    h = torch_install(stack, ci, ctx)
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


def test_custom_module_subclass_is_traced(stack, ci, ctx):
    class MyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    h = torch_install(stack, ci, ctx)
    try:
        MyNet()(torch.zeros(1, 4))
    finally:
        h.uninstall()
    assert "forward" in _names(stack.drain_closed_all())


def test_hook_exception_is_caught(stack, ci, ctx, monkeypatch, caplog):
    """A scope-push exception must not crash user code."""
    h = torch_install(stack, ci, ctx)
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


def test_no_torch_usage_after_uninstall_produces_no_spans(stack, ci, ctx):
    h = torch_install(stack, ci, ctx)
    h.uninstall()
    # With hooks gone, nothing should land in the stack.
    model = torch.nn.Linear(4, 2)
    model(torch.zeros(1, 4))
    closed = stack.drain_closed_all()
    assert closed == []


def test_many_epochs_do_not_blow_stack_depth(stack, ci, ctx):
    """Regression for PR#20 comments #1/#2: rotating epochs must actually
    leave the stack, not just be close_scope'd in place. A long run should
    stay well below ``MAX_DEPTH`` after uninstall."""
    h = torch_install(stack, ci, ctx)
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


def test_epoch_scopes_are_siblings_not_nested(stack, ci, ctx):
    """PR#20 #1: consecutive epochs must not be parent-child of each other."""
    h = torch_install(stack, ci, ctx)
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


def test_torch_does_not_yield_when_context_claimed_post_install(stack, ci):
    """Regression for CI: if ``HookContext.owned_scopes`` is empty at
    torch install time AND no other hook's callback runs to claim
    ownership, torch must still open its own epoch/step spans.

    The earlier implementation captured ``skip_epoch``/``skip_step`` as
    bools at install time. That meant a process where transformers was
    merely importable (and pre-claimed ownership at install) would
    silently lose epoch/step spans on a vanilla torch loop that never
    invoked HF ``Trainer``."""
    empty_ctx = HookContext()
    h = torch_install(stack, ci, empty_ctx)
    try:
        xs = torch.zeros(4, 4)
        ys = torch.zeros(4, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=2)
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(2):
            for xb, yb in loader:
                out = model(xb)
                ((out - yb) ** 2).mean().backward()
                opt.step()
                opt.zero_grad()
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    names = {s.name for s in closed}
    assert "epoch" in names
    assert "step" in names


def test_torch_yields_when_context_claim_appears_after_install(stack, ci):
    """The converse: a claim placed into the shared ``HookContext``
    *after* torch is installed (mirroring transformers claiming at
    ``on_train_begin`` rather than install time) must make torch yield
    on the next iteration."""
    ctx = HookContext()
    h = torch_install(stack, ci, ctx)
    try:
        xs = torch.zeros(2, 4)
        ys = torch.zeros(2, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=2)
        # Simulate another hook claiming epoch/step at runtime.
        ctx.owned_scopes["epoch"] = "transformers"
        ctx.owned_scopes["step"] = "transformers"
        for _ in loader:
            pass
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    names = {s.name for s in closed}
    assert "epoch" not in names
    assert "step" not in names


def test_torch_yields_epoch_when_owned_in_context(stack, ci):
    """When another hook has already claimed ``"epoch"``, torch must
    not open its own epoch scope — otherwise the stack gets two epoch
    spans per epoch when transformers is co-installed."""
    ctx_owned = HookContext(owned_scopes={"epoch": "transformers"})
    h = torch_install(stack, ci, ctx_owned)
    try:
        xs = torch.zeros(2, 4)
        ys = torch.zeros(2, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=2)
        for _ in range(3):
            for _ in loader:
                pass
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    epochs = [s for s in closed if s.name == "epoch"]
    assert epochs == [], f"torch emitted {len(epochs)} epoch spans when ownership was claimed"
    # data_load spans should still land — only epoch rotation is suppressed.
    assert any(s.name == "data_load" for s in closed)


def test_step_scope_wraps_forward_backward_optimizer(stack, ci, ctx):
    """One ``step`` scope per optimizer cycle, containing the per-batch
    ops as children. This is the canonical shape
    ``epoch → step → {data_load, forward, backward, optimizer_step}``
    users expect."""
    h = torch_install(stack, ci, ctx)
    try:
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        xs = torch.zeros(6, 4)
        ys = torch.zeros(6, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=2)
        for x, y in loader:
            out = model(x)
            loss = ((out - y) ** 2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    by_id = {s.id: s for s in closed}
    steps = [s for s in closed if s.name == "step"]
    assert len(steps) == 3  # 6 samples / batch 2
    # Every per-batch op nests inside its step.
    for op_name in ("data_load", "forward", "backward", "optimizer_step"):
        ops = [s for s in closed if s.name == op_name]
        assert ops, f"no {op_name} spans"
        for op in ops:
            parent = by_id.get(op.parent_id) if op.parent_id else None
            assert parent is not None and parent.name == "step", (
                f"{op_name} parent is {parent.name if parent else None!r}, expected step"
            )


def test_gradient_accumulation_produces_single_step(stack, ci, ctx):
    """Multiple forward/backward pairs between optimizer steps should
    produce ONE step span covering all of them."""
    h = torch_install(stack, ci, ctx)
    try:
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        xs = torch.zeros(4, 4)
        ys = torch.zeros(4, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=1)
        iter_count = 0
        for x, y in loader:
            out = model(x)
            loss = ((out - y) ** 2).mean()
            loss.backward()
            iter_count += 1
            if iter_count % 2 == 0:
                opt.step()
                opt.zero_grad()
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    steps = [s for s in closed if s.name == "step"]
    # 4 batches, optimizer.step every 2 → 2 step spans.
    assert len(steps) == 2


def test_torch_yields_step_when_owned_in_context(stack, ci):
    """When another hook owns ``step``, torch does not open its own."""
    ctx_owned = HookContext(owned_scopes={"step": "transformers"})
    h = torch_install(stack, ci, ctx_owned)
    try:
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        xs = torch.zeros(2, 4)
        ys = torch.zeros(2, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=2)
        for x, y in loader:
            out = model(x)
            ((out - y) ** 2).mean().backward()
            opt.step()
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    assert not [s for s in closed if s.name == "step"]


def test_user_scope_wrapping_training_loop_survives_epoch_rotation(stack, ci, ctx):
    """Epoch rotation must not pop user scopes opened above the epoch.

    Before this fix, ``_unwind_through`` popped the stack until it found
    the previous epoch, closing any scope sitting on top as collateral.
    A user scope wrapping the whole training loop would end up emitted
    (with ``end_ns`` set) after the second epoch started, even though
    the user never closed it.
    """
    h = torch_install(stack, ci, ctx)
    try:
        train_phase = stack.push("train_phase")
        assert train_phase is not None
        xs = torch.zeros(2, 4)
        ys = torch.zeros(2, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=2)
        for _ in range(3):
            for _ in loader:
                pass
        # The user scope is still open (end_ns unset) and still the
        # innermost-open scope of ours — the epoch rotation put itself
        # on top, then took itself off surgically.
        assert train_phase.end_ns is None
        # Close the user scope ourselves so we don't leave it dangling.
        stack.pop()
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    epochs = [s for s in closed if s.name == "epoch"]
    assert len(epochs) == 3
    # Every epoch has the user scope as its parent — siblings of each
    # other, children of ``train_phase``.
    for s in epochs:
        assert s.parent_id == train_phase.id, (
            f"epoch {s.index} parent is {s.parent_id!r}, expected train_phase id"
        )


def test_stopiteration_does_not_emit_data_load_span(stack, ci, ctx):
    """PR#20 #3: exhausting the iterator must not produce a trailing span."""
    h = torch_install(stack, ci, ctx)
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


def test_close_preserves_unrelated_scope_on_top(stack, ci, ctx):
    """PR#20 #4: if a user scope was opened on top of our forward span,
    _close must fall back to close_scope rather than popping the user's
    scope. The user scope must survive until they close it themselves."""
    from cirron.core.scope import get_current_scope  # local import: uses default stack

    h = torch_install(stack, ci, ctx)
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


def test_epoch_step_threshold_fallback(stack, tmp_path, ctx):
    """Without a DataLoader, optimizer steps past the threshold rotate the epoch."""
    ci = Cirron(output_dir=str(tmp_path))
    ci._profile_config = {"torch": {"epoch_steps": 2}}
    h = torch_install(stack, ci, ctx)
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
