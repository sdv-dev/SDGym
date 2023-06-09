"""Progress bars for SDGym compatible with logging and dask."""

import io
import logging
from datetime import datetime, timedelta

LOGGER = logging.getLogger(__name__)


class TqdmLogger(io.StringIO):
    """Logger for ``tqdm``."""

    _buffer = ''

    def write(self, buf):
        """Write to buffer.

        Args:
            buf (str):
                The buffer.
        """
        self._buffer = buf.strip('\r\n\t ')

    def flush(self):
        """Log the buffer."""
        LOGGER.info(self._buffer)


def progress(*futures):
    """Track progress of dask computation in a remote cluster.

    LogProgressBar is defined inside here to avoid having to import
    its dependencies if not used.
    """
    # Import distributed only when used
    from distributed.client import futures_of
    from distributed.diagnostics.progressbar import TextProgressBar

    class LogProgressBar(TextProgressBar):
        """Dask progress bar based on logging instead of stdout."""

        last = 0
        logger = logging.getLogger('distributed')

        def _draw_bar(self, remaining, total, **kwargs):
            done = total - remaining
            frac = (done / total) if total else 0

            if frac > self.last + 0.01:
                self.last = int(frac * 100) / 100
                progress_bar = '#' * int(self.width * frac)
                percent = int(100 * frac)

                time_per_task = self.elapsed / (total - remaining)
                remaining_time = timedelta(seconds=time_per_task * remaining)
                eta = datetime.utcnow() + remaining_time

                elapsed = timedelta(seconds=self.elapsed)
                msg = (  # noqa: SFS201
                    '[{0:<{1}}] | {2}/{3} ({4}%) Completed | {5} | {6} | {7}'
                ).format(
                    progress_bar, self.width, done, total, percent, elapsed, remaining_time, eta
                )
                self.logger.info(msg)
                LOGGER.info(msg)

        def _draw_stop(self, **kwargs):
            pass

    futures = futures_of(futures)
    if not isinstance(futures, (set, list)):
        futures = [futures]

    LogProgressBar(futures)
