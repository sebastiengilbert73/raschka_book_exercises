# Cf. p. 166
import logging
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def plot_projection(x, colors):  # Cf. p. 167
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.scatter(x[colors==i, 0],
                    x[colors==i, 1])
    for i in range(10):
        xtext, ytext = np.median(x[colors==i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()
        ])

def main():
    logging.info("tsne.main()")

    digits = load_digits()  # digits.keys() = dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
    fig, ax = plt.subplots(1, 4)
    for i in range(4):
        ax[i].imshow(digits.images[i], cmap='Greys')
    plt.show()

    y_digits = digits.target
    X_digits = digits.data

    tsne = TSNE(n_components=2, init='pca', random_state=123)
    X_digits_tsne = tsne.fit_transform(X_digits)

    plot_projection(X_digits_tsne, y_digits)
    plt.show()


if __name__ == '__main__':
    main()