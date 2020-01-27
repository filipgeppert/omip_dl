import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def test_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, device="cpu"):
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    results_true = np.array([])
    results_pred = np.array([])

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            results_true = np.concatenate((results_true, labels), axis=0)
            results_pred = np.concatenate((results_pred, predicted), axis=0)
            # for image, label, prediction in zip(images, labels, predicted):
            #                     plt.imshow(image[0])
            #                     plt.show()
            #     print(f"True: {label}, Predicted: {prediction}")
        accuracy = 100*correct/total

        print('Test Accuracy of the model on the test images: {} %'.format(accuracy))
    return results_true, results_pred, accuracy


def compare_and_save_model_checkpoint(state: dict, model_name: str,
                                      checkpoint_dir: str, info_dict: dict,
                                      is_best: bool = False
                                      ):
    """
    Saves model checkpoint to a given directory.

    :param state: pytorch model state dict
    :param is_best: flag to state if model performance is the best so far
    :param model_name: name of the model
    :param checkpoint_dir: path to checkpoint directory
    :param info_dict: dictionary containing model performance data
    :return: None
    """
    # Check if checkpoint dir exists, otherwise create it
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        is_best = True
    else:
        df = pd.read_csv(os.path.join(checkpoint_dir, "metrics.csv"),
                         sep=";",
                         dtype={'accuracy': float})
        if df.loc[0, 'accuracy'] < info_dict['accuracy']:
            is_best = True

    f_path = os.path.join(checkpoint_dir, f'{model_name}_checkpoint.pt')
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(checkpoint_dir, f'{model_name}_best_model.pt')
        shutil.copyfile(f_path, best_fpath)
        info_dict = {a: [d] for a, d in info_dict.items()}
        df = pd.DataFrame(info_dict)
        df.to_csv(os.path.join(checkpoint_dir, "metrics.csv"), index=False, sep=";")
