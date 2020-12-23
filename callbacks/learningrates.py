import config.autoencoderconfig as config

def poly_decay(epoch):
    """
    Sets the learning rate to decay via polynomial decay
    """
    max_epochs = config.EPOCHS
    base_lr = config.LR
    power = 1.0
    alpha = base_lr * (1 - (epoch / float(max_epochs))) ** power

    return alpha