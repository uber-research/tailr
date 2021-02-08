import numpy as np
import torch
import utils

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


def save_buffer_images(buffer, path, name, row_size=10):
    """Saves images of a given buffer in a matrix shape

    Args:
        buffer: List[torch.Tensor], list of tensors 
        path:  str, path to save the resulting figure to
        name: str, name used for the image file
        row_size: int, number of images to fit in a single row
    """
    fig = make_subplots(
        rows=len(buffer) // row_size,
        cols=row_size,
        print_grid=False,
        horizontal_spacing=0.06 / (len(buffer) // row_size),
        vertical_spacing=0.05 / row_size,
    )
    for i, tensor in enumerate(buffer):
        fig.add_trace(
            go.Heatmap(
                z=np.rot90(tensor.squeeze().T, k=1, axes=(0, 1)),
                colorscale='Greys',
                reversescale=True,
                showscale=False,
            ),
            (i // row_size) + 1,
            (i % row_size) + 1,
        )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.write_html(f'{path}/{name}.html')


def save_images_per_class(n_classes,
                          gen_buffer_images,
                          gen_buffer_class_labels,
                          gen_image_count,
                          n_y_active_np,
                          save_path):

    for i in range(n_classes):
        class_idx = gen_buffer_class_labels == i
        with open(f'{save_path}/class_counts_log.txt', 'a+') as f:
            f.write(f'Class {i}: {class_idx.sum()} ')
        if class_idx.sum() < gen_image_count and class_idx.sum() > 0:
            n_gen = int(class_idx.sum() - class_idx.sum() % 10)
            buffer = gen_buffer_images[class_idx][:n_gen]
            if len(buffer) > 0:
                save_buffer_images(
                    buffer, path=save_path, name=f'active_{n_y_active_np}_class_{i}_badly_represented', row_size=min(10, len(buffer)))
        elif class_idx.sum() >= gen_image_count:
            buffer = gen_buffer_images[class_idx][:gen_image_count]
            save_buffer_images(
                buffer, path=save_path, name=f'active_{n_y_active_np}_class_{i}', row_size=min(10, len(buffer)))

    with open(f'{save_path}/class_counts_log.txt', 'a+') as f:
        f.write(f'\n\n')


def save_images_per_cluster(sess,
                            n_y,
                            y_gen_image,
                            gen_images,
                            gen_image_count,
                            n_y_active_np,
                            save_path):
    for cluster_id in range(n_y_active_np):
        y_gen_posterior_vals = np.zeros(
            (gen_image_count, n_y))
        y_gen_posterior_vals[:, cluster_id] = 1
        buffer = sess.run(gen_images, feed_dict={
            y_gen_image: y_gen_posterior_vals})
        save_buffer_images(
            buffer, path=save_path, name=f'active_{n_y_active_np}_clusterid_{cluster_id}', row_size=min(10, len(buffer)))
        del buffer
