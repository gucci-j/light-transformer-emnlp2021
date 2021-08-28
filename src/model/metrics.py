from transformers import EvalPrediction

def compute_metrics_fn_for_mtl(p: EvalPrediction):
    """
    Dummy `compute_metrics_fn` function to export loss information to Tensorboard.
    This function will not compute any evaluation metrics.

    References:
        https://huggingface.co/transformers/internal/trainer_utils.html#transformers.EvalPrediction
    """
    # get loss tensors
    cls_loss = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    reg_loss = p.predictions[1] if isinstance(p.predictions, tuple) else p.predictions
    
    # return dict
    return {"cls_loss": cls_loss, "reg_loss": reg_loss}


def compute_metrics_fn_for_shuffle_random(p: EvalPrediction):
    """
    Dummy `compute_metrics_fn` function to export loss information to Tensorboard.
    This function will not compute any evaluation metrics.

    References:
        https://huggingface.co/transformers/internal/trainer_utils.html#transformers.EvalPrediction
    """
    # get loss tensors
    shuffle_loss = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    random_loss = p.predictions[1] if isinstance(p.predictions, tuple) else p.predictions
    
    # return dict
    return {"shuffle_loss": shuffle_loss, "random_loss": random_loss}