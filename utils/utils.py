from stable_baselines3.common.callbacks import BaseCallback
from statistics import mean


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        illegal = []
        won = []
        hit = []
        stand = []
        double = []
        surrender = []
        split = []
        return_ = []

        for idx, info in enumerate(self.locals["infos"]):
            if info.get("episode") is not None:
                episode = info["episode"]
                illegal.append(episode["illegal"])
                won.append(episode["won"])
                hit.append(episode["hit"])
                stand.append(episode["stand"])
                double.append(episode["double"])
                surrender.append(episode["surrender"])
                split.append(episode["split"])
                return_.append(episode["return"])

        if len(illegal) > 0:
            self.logger.record("action_log/illegal", mean(illegal))
            self.logger.record("action_log/won", mean(won))
            self.logger.record("action_log/hit", mean(hit))
            self.logger.record("action_log/stand", mean(stand))
            self.logger.record("action_log/double", mean(double))
            self.logger.record("action_log/surrender", mean(surrender))
            self.logger.record("action_log/split", mean(split))
            self.logger.record("action_log/return", mean(return_))

        return True
