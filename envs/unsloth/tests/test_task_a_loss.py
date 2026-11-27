"""Opt-in CUDA loss-contract smoke with tiny tensors, no model weights loaded."""

import os

import pytest


@pytest.mark.parametrize("loss_path", ["logits", "chunked"])
def test_unsloth_default_reduction_matches_shifted_nll(tmp_path, loss_path):
    if os.environ.get("LM_TRAINER_CUDA_SMOKE") != "1":
        pytest.skip("Set LM_TRAINER_CUDA_SMOKE=1 in a fresh Unsloth interpreter")
    from lm_trainer.train.bootstrap import initialize_backend

    # Import order is part of the test: initialize Unsloth before its peers.
    initialize_backend({"backend": "unsloth", "checkpoint_dir": str(tmp_path)})
    from types import SimpleNamespace

    import torch
    from accelerate import Accelerator
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss
    from unsloth_zoo.fused_losses import unsloth_fused_ce_loss

    from lm_trainer.train.engine import update_window

    assert torch.cuda.is_available()
    torch.manual_seed(7)
    size = 24

    class TinyFusedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Embedding(size, size, device="cuda")
            self.register_buffer("lm_head", torch.eye(size, device="cuda"))

        def forward(self, input_ids, attention_mask, labels):
            assert attention_mask.shape == labels.shape
            hidden = self.projection(input_ids)
            if loss_path == "chunked":
                # Same raw labels and internal shift/default reduction as the
                # standard Unsloth no-logits causal forward.
                loss = unsloth_fused_ce_loss(None, hidden, self.lm_head, None, labels)
            else:
                shift = torch.nn.functional.pad(labels[:, 1:], (0, 1), value=-100)
                loss = fast_cross_entropy_loss(hidden, shift)
            return SimpleNamespace(loss=loss)

    model = TinyFusedModel()
    reference = TinyFusedModel()
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    accelerator = Accelerator(gradient_accumulation_steps=1)

    def batch(ids, labels):
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(ids).ne(0).long(),
        }

    rows = [
        batch([[4, 5, 6]], [[4, -100, 6]]),
        batch([[3, 2, 3, 4, 5]], [[3, 2, 3, -100, 5]]),
    ]
    combined = batch(
        [[4, 5, 6, 0, 0], [3, 2, 3, 4, 5]],
        [[4, -100, 6, -100, -100], [3, 2, 3, -100, 5]],
    )
    logits = reference.projection(combined["input_ids"].cuda())
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, size), combined["labels"][:, 1:].reshape(-1).cuda()
    )
    expected.backward()
    loss, targets = update_window(model, optimizer, scheduler, accelerator, rows)
    assert targets == 4
    assert loss == pytest.approx(expected.item(), abs=2e-6)
    torch.testing.assert_close(
        model.projection.weight,
        reference.projection.weight - 0.1 * reference.projection.weight.grad,
        atol=3e-7,
        rtol=3e-6,
    )
