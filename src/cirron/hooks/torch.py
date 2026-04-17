"""PyTorch hooks — stub for SDK-13.

Per spec §4.8: forward-pass (``nn.Module.__call__``), backward-pass (autograd
``Tensor.backward``), optimizer step (``optim.Optimizer.step``), DataLoader
iteration, CUDA timing. Installed automatically by ``ci.profile()`` when
``torch`` is importable.
"""
