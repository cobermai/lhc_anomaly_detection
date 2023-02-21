from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def plot_confusion_signals(X_test, y_test, y_pred, all_labels, output_path, n_split=0):
    fig, ax = plt.subplots(len(all_labels), len(all_labels), figsize=(len(all_labels)*5, len(all_labels)*5))

    for idx_pred, label_pred in enumerate(all_labels):
        for idx_true, label_true in enumerate(all_labels):
            bool = (y_test == idx_true) & (y_pred == idx_pred)

            if sum(bool) > 0:
                if label_pred == label_true:
                    ax[idx_true, idx_pred].plot(X_test[bool].T, c="g", alpha=0.3)
                else:
                    ax[idx_true, idx_pred].plot(X_test[bool].T, c="r", alpha=0.3)

            ax[idx_true, idx_pred].set_ylim((X_test.min(), X_test.max()))
            ax[idx_true, idx_pred].set_title(f"{label_true} classified {label_pred} ({sum(bool)})", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / f"{n_split}_confusion_signals")


def plot_confusion_matrix(y_test_argmax, y_pred_argmax, all_labels, output_path, n_split=0):
    # Plot confusion matrix
    result = classification_report(y_test_argmax, y_pred_argmax, target_names=all_labels)

    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    cm = confusion_matrix(y_test_argmax, y_pred_argmax)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels).plot(ax=ax[0])
    ax[0].set_title(f"Split {n_split}")
    ax[0].set_xticklabels(all_labels, rotation=45)
    ax[1].text(0, 1,
               result,
               horizontalalignment='left',
               verticalalignment='top')
    ax[1].set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path / f"{n_split}_confusion_matrix")