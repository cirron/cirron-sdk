"""PyTorch hooks — unit tests.

Skipped in environments without ``torch`` so the core CI path (no
frameworks installed) stays green. When ``torch`` is available, we
exercise each hook target independently and confirm ``uninstall()``
restores the originals.
"""

from __future__ import annotations

import weakref

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


class _CountingModel(torch.nn.Module):
    """Linear model that counts how often ``named_parameters`` is walked."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)
        self.named_parameters_calls = 0

    def forward(self, x):
        return self.fc(x)

    def named_parameters(self, *args, **kwargs):
        self.named_parameters_calls += 1
        return super().named_parameters(*args, **kwargs)


def _drive_step(model, opt):
    """Run one forward/backward/optimizer cycle on ``model``."""
    out = model(torch.zeros(1, 2))
    out.sum().backward()
    opt.step()


def test_grad_stash_caches_named_parameters(stack, ci, ctx):
    """The per-step grad stash walks named_parameters() once, not per step."""
    import cirron as public_ci

    model = _CountingModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    # nn.Module.parameters() is itself implemented over named_parameters(),
    # so building the optimizer already walks it. Measure from here.
    base = model.named_parameters_calls

    public_ci.watch(model)
    h = torch_install(stack, ci, ctx)
    try:
        _drive_step(model, opt)
        after_first = model.named_parameters_calls - base
        _drive_step(model, opt)
        after_second = model.named_parameters_calls - base
        # Read the counter before uninstall: uninstall snapshots the final
        # epoch, which legitimately walks the parameters again.
        assert after_first == 1, "first step should populate the cache"
        assert after_second == 1, f"second step re-walked the model ({after_second} walks)"
    finally:
        h.uninstall()
        public_ci.watch(None)


def test_grad_stash_snapshots_the_latest_step(stack, ci, ctx):
    """Caching the parameter list must not cache the gradients themselves.

    Drives two steps with different inputs, so the final step's grads are
    distinguishable, then checks the epoch-boundary snapshot reports the
    final step's values rather than the first step's.
    """
    import cirron as public_ci
    from cirron.core.snapshot_buffer import (
        _reset_default_for_tests,
        get_default_snapshot_buffer,
    )

    _reset_default_for_tests()
    model = _CountingModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0 keeps weights fixed
    public_ci.watch(model)
    h = torch_install(stack, ci, ctx)
    try:
        # Iterating a loader is what opens an epoch scope; without one there
        # is no span for uninstall to attach the final snapshot to.
        xs = torch.zeros(1, 2)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs, torch.zeros(1, 1)), batch_size=1
        )
        for _ in loader:
            pass

        # Step 1: small input, small grads.
        out = model(torch.ones(1, 2))
        out.sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        # Step 2: input scaled by 100, so the bias grad is unchanged but
        # the weight grads are two orders of magnitude larger.
        out = model(torch.ones(1, 2) * 100.0)
        out.sum().backward()
        expected = model.fc.weight.grad.mean().item()
        opt.step()
        opt.zero_grad(set_to_none=True)
    finally:
        h.uninstall()  # uninstall snapshots the open epoch
        public_ci.watch(None)

    records = get_default_snapshot_buffer().drain()
    grads = {r.tensor_name: r for r in records if r.tensor_name.endswith(".grad")}
    assert "fc.weight.grad" in grads, f"no weight grad snapshot in {sorted(grads)}"
    got = grads["fc.weight.grad"].stats["mean"]
    assert got == pytest.approx(expected), (
        f"snapshot reports {got}, expected the final step's grads ({expected})"
    )
    _reset_default_for_tests()


def test_grad_stash_rebuilds_on_new_model(stack, ci, ctx):
    """Watching a different model invalidates the cached parameter pairs."""
    import cirron as public_ci

    model_a = _CountingModel()
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.01)
    model_b = _CountingModel()
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)
    # Optimizer construction already walked each model once (see above).
    base_a = model_a.named_parameters_calls
    base_b = model_b.named_parameters_calls

    public_ci.watch(model_a)
    h = torch_install(stack, ci, ctx)
    try:
        _drive_step(model_a, opt_a)
        _drive_step(model_a, opt_a)
        assert model_a.named_parameters_calls - base_a == 1

        public_ci.watch(model_b)
        _drive_step(model_b, opt_b)
        _drive_step(model_b, opt_b)
        assert model_b.named_parameters_calls - base_b == 1, (
            "cache did not rebuild for the new model"
        )
        # A's cache entry was replaced, not consulted again.
        assert model_a.named_parameters_calls - base_a == 1
    finally:
        h.uninstall()
        public_ci.watch(None)


def test_grad_stash_releases_cache_when_model_cleared(stack, ci, ctx):
    """ci.watch(None) must drop the cached parameter references.

    The cache holds every parameter tensor strongly, so retaining it past
    the watched model would pin a full set of weights for the life of the
    process.
    """
    import gc

    import cirron as public_ci

    model = _CountingModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    public_ci.watch(model)
    h = torch_install(stack, ci, ctx)
    try:
        _drive_step(model, opt)
        public_ci.watch(None)
        # Next step observes no watched model and releases the cache.
        _drive_step(model, opt)
        weight_ref = weakref.ref(model.fc.weight)
        del model, opt
        gc.collect()
        assert weight_ref() is None, "cached parameter pairs kept the weights alive"
    finally:
        h.uninstall()
        public_ci.watch(None)


def test_should_reap_cadence():
    """Reaping is periodic, with a backstop against unbounded growth."""
    from cirron.hooks._torch_impl import _REAP_BACKSTOP, _REAP_INTERVAL, _should_reap

    # Not every op close pays for a scan.
    assert not _should_reap(1, 0)
    assert not _should_reap(_REAP_INTERVAL - 1, 0)
    # The periodic reap fires on the interval.
    assert _should_reap(_REAP_INTERVAL, 0)
    assert _should_reap(_REAP_INTERVAL * 2, 0)
    # The backstop fires regardless of where we are in the interval, so a
    # workload whose events resolve slowly still gets drained.
    assert _should_reap(_REAP_INTERVAL + 1, _REAP_BACKSTOP)
    assert _should_reap(_REAP_INTERVAL + 1, _REAP_BACKSTOP + 500)
    assert not _should_reap(_REAP_INTERVAL + 1, _REAP_BACKSTOP - 1)


def test_cuda_pending_starts_with_empty_pool():
    """The event pool and op counter start clean on every install."""
    from cirron.hooks._torch_impl import _CudaPending

    pending = _CudaPending()
    assert pending.items == []
    assert pending.pool == []
    assert pending.ops == 0


def test_drain_recycles_events_after_reading_elapsed_time():
    """Both events of a resolved pair return to the pool, exactly once."""
    from cirron.hooks._torch_impl import _CudaPending, _drain_cuda

    class _Scope:
        gpu_ns = None

    class _Event:
        def __init__(self, ready=True):
            self._ready = ready

        def query(self):
            return self._ready

        def elapsed_time(self, other):
            return 2.0

    scope_obj = _Scope()
    start_ev, end_ev = _Event(), _Event()
    pending = _CudaPending()
    pending.items.append((scope_obj, start_ev, end_ev))

    _drain_cuda(pending, force=False)

    assert scope_obj.gpu_ns == 2_000_000
    assert pending.items == []
    assert pending.pool == [start_ev, end_ev]


def test_drain_keeps_unresolved_pairs_out_of_the_pool():
    """An event still in flight is neither read nor recycled."""
    from cirron.hooks._torch_impl import _CudaPending, _drain_cuda

    class _Scope:
        gpu_ns = None

    class _PendingEvent:
        def query(self):
            return False

        def elapsed_time(self, other):
            raise AssertionError("elapsed_time read before the event resolved")

    scope_obj = _Scope()
    start_ev, end_ev = _PendingEvent(), _PendingEvent()
    pending = _CudaPending()
    pending.items.append((scope_obj, start_ev, end_ev))

    _drain_cuda(pending, force=False)

    assert scope_obj.gpu_ns is None
    assert len(pending.items) == 1, "unresolved pair must stay pending"
    assert pending.pool == [], "an event still in flight must never be reused"


def test_drain_does_not_recycle_events_that_failed():
    """A pair whose elapsed_time raised is dropped without being reused."""
    from cirron.hooks._torch_impl import _CudaPending, _drain_cuda

    class _Scope:
        gpu_ns = None

    class _BrokenEvent:
        def query(self):
            return True

        def elapsed_time(self, other):
            raise RuntimeError("device error")

    scope_obj = _Scope()
    pending = _CudaPending()
    pending.items.append((scope_obj, _BrokenEvent(), _BrokenEvent()))

    _drain_cuda(pending, force=False)

    assert scope_obj.gpu_ns is None
    assert pending.items == [], "the failed pair must not stay pending forever"
    assert pending.pool == [], "a misbehaving event must not be handed to the next span"


def test_event_pool_is_capped():
    """The pool stops growing at its cap so a backlog cannot hoard events."""
    from cirron.hooks._torch_impl import _EVENT_POOL_CAP, _CudaPending, _drain_cuda

    class _Scope:
        gpu_ns = None

    class _Event:
        def query(self):
            return True

        def elapsed_time(self, other):
            return 1.0

    pending = _CudaPending()
    for _ in range(_EVENT_POOL_CAP):  # two events each, so well past the cap
        pending.items.append((_Scope(), _Event(), _Event()))

    _drain_cuda(pending, force=False)

    assert pending.items == []
    assert len(pending.pool) <= _EVENT_POOL_CAP


def test_param_cache_released_when_model_is_collected_without_further_steps(stack, ci, ctx):
    """The cache must not outlive the model it describes.

    The opportunistic clear inside the stash only runs on a *later*
    optimizer step. A run that simply stops stepping (training finished,
    model dropped) would otherwise pin a full set of parameter tensors
    until uninstall, so collection itself has to release them.
    """
    import gc

    import cirron as public_ci

    h = torch_install(stack, ci, ctx)
    try:
        model = _CountingModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        public_ci.watch(model)
        _drive_step(model, opt)  # populates the cache

        weight_ref = weakref.ref(model.fc.weight)
        assert weight_ref() is not None

        # Drop the model and never step again.
        del model, opt
        gc.collect()

        assert weight_ref() is None, (
            "parameter tensors survived collection of the model; the cache pinned them"
        )
    finally:
        h.uninstall()
        public_ci.watch(None)


def test_param_cache_survives_model_swap(stack, ci, ctx):
    """Collecting a replaced model must not clear the new model's cache."""
    import gc

    import cirron as public_ci

    h = torch_install(stack, ci, ctx)
    try:
        model_a = _CountingModel()
        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.01)
        public_ci.watch(model_a)
        _drive_step(model_a, opt_a)

        model_b = _CountingModel()
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)
        public_ci.watch(model_b)
        _drive_step(model_b, opt_b)
        base_b = model_b.named_parameters_calls

        # A's finalizer must have been detached when B took over; if it
        # fires now it would wrongly clear B's cache.
        del model_a, opt_a
        gc.collect()

        _drive_step(model_b, opt_b)
        assert model_b.named_parameters_calls == base_b, (
            "collecting the previous model invalidated the current model's cache"
        )
    finally:
        h.uninstall()
        public_ci.watch(None)


# deferred close: CUDA-timed scopes wait for their events
#
# CI has no GPU, so pending_cuda is None in a real install and these paths
# never run there. _drain_cuda only ever calls query() and elapsed_time()
# on an event, so the holder is driven directly with fakes instead. No
# torch.cuda internals are mocked.


class _FakeEvent:
    """Duck-typed stand-in for torch.cuda.Event."""

    def __init__(self, ready=True, elapsed_ms=2.0, fail=False):
        self.ready = ready
        self.elapsed_ms = elapsed_ms
        self.fail = fail

    def query(self):
        return self.ready

    def elapsed_time(self, other):
        if self.fail:
            raise RuntimeError("device error")
        return self.elapsed_ms


def _held_scope(stack, pending, start_ev, end_ev):
    """Push a scope, hold it the way the CUDA path does, and queue its pair."""
    scope_obj = stack.push("forward")
    stack.finalize_deferred(scope_obj)
    pending.items.append((scope_obj, start_ev, end_ev))
    return scope_obj


def test_held_scope_is_not_drainable_until_its_event_resolves():
    """The race this closes: a closed scope must not reach the flush thread
    before its gpu_ns has been written."""
    from cirron.hooks._torch_impl import _CudaPending, _drain_cuda

    stack = ScopeStack()
    pending = _CudaPending()
    pending.scope_stack = stack
    start_ev, end_ev = _FakeEvent(ready=False), _FakeEvent(ready=False)
    scope_obj = _held_scope(stack, pending, start_ev, end_ev)

    _drain_cuda(pending, force=False)
    assert stack.drain_closed_all() == [], "scope drained before its GPU timing was known"
    assert scope_obj.gpu_ns is None
    assert len(pending.items) == 1

    # The kernel finishes; now it resolves and becomes drainable.
    start_ev.ready = end_ev.ready = True
    _drain_cuda(pending, force=False)

    drained = stack.drain_closed_all()
    assert drained == [scope_obj]
    assert scope_obj.gpu_ns == 2_000_000
    assert pending.items == []


def test_held_scope_is_emitted_exactly_once():
    """A second drain must not emit the scope again."""
    from cirron.hooks._torch_impl import _CudaPending, _drain_cuda

    stack = ScopeStack()
    pending = _CudaPending()
    pending.scope_stack = stack
    scope_obj = _held_scope(stack, pending, _FakeEvent(), _FakeEvent())

    _drain_cuda(pending, force=False)
    _drain_cuda(pending, force=False)

    assert stack.drain_closed_all() == [scope_obj], "scope was emitted more than once"


def test_held_scope_is_emitted_even_when_elapsed_time_fails():
    """Losing GPU timing must not lose the span itself."""
    from cirron.hooks._torch_impl import _CudaPending, _drain_cuda

    stack = ScopeStack()
    pending = _CudaPending()
    pending.scope_stack = stack
    scope_obj = _held_scope(stack, pending, _FakeEvent(fail=True), _FakeEvent())

    _drain_cuda(pending, force=False)

    drained = stack.drain_closed_all()
    assert drained == [scope_obj], "a span was dropped because its GPU timing failed"
    assert scope_obj.gpu_ns is None
    assert scope_obj.end_ns is not None, "the span should still carry wall-clock timing"
    assert pending.items == [], "the failed pair must not stay pending forever"


def test_forced_drain_emits_every_held_scope():
    """Uninstall forces a drain; nothing may be left held afterwards."""
    from cirron.hooks._torch_impl import _CudaPending, _drain_cuda

    stack = ScopeStack()
    pending = _CudaPending()
    pending.scope_stack = stack
    held = [
        _held_scope(stack, pending, _FakeEvent(ready=False), _FakeEvent(ready=False))
        for _ in range(5)
    ]

    _drain_cuda(pending, force=True)

    drained = stack.drain_closed_all()
    assert len(drained) == len(held), f"{len(held) - len(drained)} held scopes were lost"
    assert {id(s) for s in drained} == {id(s) for s in held}
    assert pending.items == []


def test_drain_without_a_scope_stack_does_not_raise():
    """Defensive: a holder with no stack reference degrades quietly."""
    from cirron.hooks._torch_impl import _CudaPending, _drain_cuda

    class _Scope:
        gpu_ns = None

    pending = _CudaPending()  # scope_stack left as None
    pending.items.append((_Scope(), _FakeEvent(), _FakeEvent()))

    _drain_cuda(pending, force=True)

    assert pending.items == []
