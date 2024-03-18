from mmdet.core.evaluation import EvalHook
from mmcv.runner import TextLoggerHook


class TrainValHook(EvalHook):
    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        for hook in runner._hooks:
            if isinstance(hook, TextLoggerHook) and runner.meta is not None:
                hook.after_train_iter(runner)
        # runner.log_buffer.clear()
        if self.by_epoch and self._should_evaluate(runner):
            self._do_evaluate(runner)

    # def after_val_epoch(self, runner):
    #             # print(runner.meta)
    #             # hook._dump_log(runner.meta, runner)
    #     runner.log_buffer.clear()
    #     self.after_train_epoch(runner)
