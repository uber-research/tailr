import numpy as np
from torch.cuda import FloatTensor as CUDATensor
from visdom import Visdom

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

_WINDOW_CASH = {}


class visualization:

    def __init__(self, env, server='http://localhost', port=8097):
        self.vis = Visdom(env=env, server=server, port=port)

    def visualize_image(self, tensor, name, label=None, env='main', w=250, h=250,
                        update_window_without_label=False):

        tensor = tensor.cpu() if isinstance(tensor, CUDATensor) else tensor
        title = name + ('-{}'.format(label) if label is not None else '')

        _WINDOW_CASH[title] = self.vis.image(
            tensor.numpy(), win=_WINDOW_CASH.get(title),
            opts=dict(title=title, width=w, height=h)
        )

        # This is useful when you want to maintain the most recent images.
        if update_window_without_label:
            _WINDOW_CASH[name] = self.vis.image(
                tensor.numpy(), win=_WINDOW_CASH.get(name),
                opts=dict(title=name, width=w, height=h)
            )

    def visualize_images(self, tensor, name, label=None, env='main', w=400, h=400,
                         update_window_without_label=False):
        tensor = tensor.cpu() if isinstance(tensor, CUDATensor) else tensor
        title = name + ('-{}'.format(label) if label is not None else '')

        _WINDOW_CASH[title] = self.vis.images(
            tensor.numpy(), win=_WINDOW_CASH.get(title), nrow=6,
            opts=dict(title=title, width=w, height=h)
        )

        # This is useful when you want to maintain the most recent images.
        if update_window_without_label:
            _WINDOW_CASH[name] = self.vis.images(
                tensor.numpy(), win=_WINDOW_CASH.get(name), nrow=6,
                opts=dict(title=name, width=w, height=h)
            )

    def visualize_kernel(self, kernel, name, label=None, env='main', w=250, h=250,
                         update_window_without_label=False, compress_tensor=False):
        # Do not visualize kernels that does not exists.
        if kernel is None:
            return

        assert len(kernel.size()) in (2, 4)
        title = name + ('-{}'.format(label) if label is not None else '')
        kernel = kernel.cpu() if isinstance(kernel, CUDATensor) else kernel
        kernel_norm = kernel if len(kernel.size()) == 2 else (
            (kernel**2).mean(-1).mean(-1) if compress_tensor else
            kernel.view(
                kernel.size()[0] * kernel.size()[2],
                kernel.size()[1] * kernel.size()[3],
            )
        )
        kernel_norm = kernel_norm.abs()

        visualized = (
            (kernel_norm - kernel_norm.min()) /
            (kernel_norm.max() - kernel_norm.min())
        ).numpy()

        _WINDOW_CASH[title] = self.vis.image(
            visualized, win=_WINDOW_CASH.get(title),
            opts=dict(title=title, width=w, height=h)
        )

        # This is useful when you want to maintain the most recent images.
        if update_window_without_label:
            _WINDOW_CASH[name] = self.vis.image(
                visualized, win=_WINDOW_CASH.get(name),
                opts=dict(title=name, width=w, height=h)
            )

    def visualize_scalar(self, scalar, name, iteration, env='main'):
        self.visualize_scalars(
            [scalar] if isinstance(scalar, float) or len(
                scalar) == 1 else scalar,
            [name], name, iteration, env=env
        )

    def visualize_scalars(self, scalars, names, title, iteration, env='main'):
        assert len(scalars) == len(names)
        # Convert scalar tensors to numpy arrays.
        scalars, names = list(scalars), list(names)
        scalars = [s.cpu() if isinstance(s, CUDATensor)
                   else s for s in scalars]
        scalars = [s.numpy() if hasattr(s, 'numpy') else np.array([s]) for s in
                   scalars]
        multi = len(scalars) > 1
        num = len(scalars)

        options = dict(
            fillarea=True,
            legend=names,
            width=400,
            height=400,
            xlabel='Iterations',
            ylabel=title,
            title=title,
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
        )

        X = (
            np.column_stack(np.array([iteration] * num)) if multi else
            np.array([iteration] * num)
        )
        Y = np.column_stack(scalars) if multi else scalars[0]

        if title in _WINDOW_CASH:
            # Deprecated
            #_vis(env).updateTrace(X=X, Y=Y, win=_WINDOW_CASH[title], opts=options)
            self.vis.line(
                X=X, Y=Y, win=_WINDOW_CASH[title], opts=options, update='append')
        else:
            _WINDOW_CASH[title] = self.vis.line(X=X, Y=Y, opts=options)


def save_buffer_images(buffer, path, name, row_size=10):
    """Saves images of a given buffer in a matrix shape

    Args:
        buffer: list of tensors, List[torch.Tensor]
        path: path to save the resulting figure to, str
        name: name used for the image file, str
        row_size: number of images to fit in a single row
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
