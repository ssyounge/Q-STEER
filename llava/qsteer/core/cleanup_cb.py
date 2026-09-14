from transformers import TrainerCallback

from .route_context import QSTEERContext


class QSTEERClearCallback(TrainerCallback):
    """Clear thread-local QSTEER context after each completed optimizer step."""

    def on_step_end(self, args, state, control, **kwargs):
        QSTEERContext.clear()

    def on_train_end(self, args, state, control, **kwargs):
        QSTEERContext.clear()
