# Module for starting up a Tensorboard instance

import time
import click
import tensorboard


@click.command()
@click.argument(
    'logdir',
    default='./figs/',
    type=click.STRING,
)
def main(logdir: str) -> None:
    """ Launch TensorBoard 2.0 """

    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--bind_all'])
    url = tb.launch()
    print(f'Tensorboard is available from {url}')
    wait_for_quitting()


def wait_for_quitting():
    """Waiting function"""

    print(f'Press ctrl-c to quit Tensorboard')
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        exit(0)


if __name__ == '__main__':
    main()
