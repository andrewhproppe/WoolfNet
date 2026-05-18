"""
General util functions
"""

import atexit
import time


class RuntimeStatistics:
    """
    Class used to collect runtime statistics over the course of a programs
    lifetime in `ml-models`
    """

    def __init__(self):
        self.checkpoint = time.time()
        atexit.register(self.print_stats)

    def get_elapsed_time(self, update_checkpoint: bool = False) -> float:
        """
        Calculate the time elapsed between the internal checkpoint and
        the moment this function is called.

        Parameters
        ----------
        update_checkpoint : bool, optional
            Update the underlying time checkpoint used to calculate the current
            runtime, by default False

        Returns
        -------
        float
            The current run time measured in seconds
        """
        runtime = time.time() - self.checkpoint
        if update_checkpoint:
            self.checkpoint = time.time()
        return runtime

    def print_stats(self) -> None:
        """
        Output diagnostic information whenever a program within `ml-models`
        has completed running.
        """
        header = "\nProgram Diagnostics"
        outputs = {"Runtime": f"{self.get_elapsed_time():.2f} sec"}
        max_key = max(len(k) for k in outputs.keys())
        max_value = max(len(v) for v in outputs.values())
        length = max_key + max_value + 4

        print(header)
        print("-" * length)
        for k, v in outputs.items():
            print(f"{k:<{max_key}}    {v:<{max_value + 4}}")
        print("-" * length)
