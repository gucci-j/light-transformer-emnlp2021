from transformers import TrainerCallback
import time
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

class LoggingCallback(TrainerCallback):
    """Callback class for saving weights periodically.
    """

    def __init__(self, save_interval: float):
        """
        Args:
            save_interval (float): An interval to save weights in seconds.  
        """
        self.save_interval = save_interval
        self.start_time = time.time()
        self.save_counter = 1

    
    def on_log(self, args, state, control, logs=None, **kwargs):
        current_duration = time.time() - self.start_time
        if (current_duration // (self.save_interval * self.save_counter)) >= 1:
            logger.info(f'Save weights at {state.global_step} steps trained for '
                        f'{self.save_interval} * {self.save_counter} seconds!')
            self.save_counter += 1
            control.should_save = True
            

class EarlyStoppingCallback(TrainerCallback):
    """Callback class for enabling a model to early stop its training.
    Reference: https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback
    """

    def __init__(self, patience: int, metric_name: str, objective_type: str, verbose: bool=True):
        """
        Args:
            patience (int): Patience value for early stopping.
                            Actual resulting value should be (logging_steps * patience).
            metric_name (str): Which metric to watch? The name should be based on `metrics`.
            objective_type (str): `maximize` or `minimize`.
            verbose (bool): Whether to show status. (default=True)
        """
        self.patience = patience
        self.metric_name = metric_name
        self.objective_type = objective_type
        self.verbose = verbose
        self.best_metric = None
        self.counter = 0
        

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Event called after the evaluation phase."""
        
        # score calculation
        score = metrics.pop(self.metric_name, None)
        if score is None: # illegal name is given
            return
        if self.objective_type == "minimize": # such as loss
            score = -score

        # score evaluation
        if self.best_metric is None: 
            # init
            self.best_metric = score

        elif score < self.best_metric: 
            # not improved
            self.counter += 1
            if self.verbose:
                logger.info(f"Early stopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    logger.info(f"Early stopping applied at {state.global_step} steps!")
                control.should_training_stop = True

        else: # improved
            self.counter = 0
            if self.verbose:
                logger.info(f"Score improved: {self.best_metric:.3f} -> {score:.3f}")
            self.best_metric = score
            state.best_model_checkpoint = args.output_dir + "/checkpoint-" + str(state.global_step)
            control.should_save = True


class EarlyStopping:
    """Early stopping callback for SQuAD."""
    def __init__(self, patience: int, metric_name: str, objective_type: str, verbose: bool=True):
        """
        Args:
            patience (int): Patience value for early stopping.
                            Actual resulting value should be (logging_steps * patience).
            metric_name (str): Which metric to watch? The name should be based on `metrics`.
            objective_type (str): `maximize` or `minimize`.
            verbose (bool): Whether to show status. (default=True)
        """

        self.patience = patience
        self.metric_name = metric_name
        self.objective_type = objective_type
        self.verbose = verbose
        self.best_metric = None
        self.counter = 0
        self.should_stop_training = False
        self.should_save = False


    def __call__(self, results: dict, global_step: int=None):
        # score calculation
        score = results.pop(self.metric_name, None)
        if score is None: # illegal name is given
            return
        if self.objective_type == "minimize": # such as loss
            score = -score

        # score evaluation
        if self.best_metric is None: 
            # init
            self.best_metric = score
            self.should_save = True

        elif score < self.best_metric: 
            # not improved
            self.counter += 1
            self.should_save = False
            if self.verbose:
                logger.info(f"[{global_step} steps]: Early stopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    logger.info(f"[{global_step} steps]: Early stopping applied!")
                self.should_stop_training = True

        else: # improved
            self.counter = 0
            if self.verbose:
                logger.info(f"[{global_step} steps]: Score improved: {self.best_metric:.3f} -> {score:.3f}")
            self.best_metric = score
            self.should_save = True
