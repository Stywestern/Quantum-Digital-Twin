from abc import ABC, abstractmethod

class BaseSolver(ABC):
    def __init__(self, max_time=0):
        self.max_time = max_time

    @abstractmethod
    def solve(self, net):
        """
        Executes the solver on the provided Pandapower network.
        Must return: (solution: dict, metadata: dict)
        """
        pass