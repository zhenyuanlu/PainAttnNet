import torch


def _save_checkpoint(model, optimizer, epoch, mnt_best, config, checkpoint_dir, logger, save_best = True):
    """
    Saving checkpoints

    :param model: model to be saved
    :param optimizer: optimizer to be saved
    :param epoch: current epoch number
    :param mnt_best: metric used to monitor the best model
    :param config: configuration
    :param checkpoint_dir: directory where the checkpoint will be saved
    :param logger: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    arch = type(model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'monitor_best': mnt_best,
        'config': config
    }
    filename = str(checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
    torch.save(state, filename)
    logger.info("Saving checkpoint: {} ...".format(filename))
    if save_best:
        best_path = str(checkpoint_dir / 'model_best.pth')
        torch.save(state, best_path)
        logger.info("Saving current best: model_best.pth ...")

