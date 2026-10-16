from dataclasses import dataclass,field
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class Task:
    id: int
    layer: int
    points: np.ndarray
    robots: tuple[int, ...]
    residual_fill: bool = False
    _length: float = field(init=False,repr=False)

    def __post_init__(self):
        points=np.array(self.points,dtype=float,copy=True)
        points.setflags(write=False)
        object.__setattr__(self,'points',points)
        object.__setattr__(self,'robots',tuple(self.robots))
        object.__setattr__(self,'_length',float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()))

    @property
    def length(self):
        return self._length


@dataclass(frozen=True)
class Action:
    task: Task
    robot: int
    reverse: bool


class Scheduler(Protocol):
    def select(self, tasks: list[Task], state, config: dict) -> Action:
        """Choose a feasible task/robot/direction from the current layer.

        REINFORCE can implement this interface. The common execution engine
        still enforces reach, layer precedence, collision, and CSV semantics.
        """
        ...


class ParallelScheduler(Protocol):
    def plan_layer(self, tasks: list[Task], engine, release: float) -> None:
        """Assign/order ready tasks and commit collision-resolved robot tours.

        A learned policy may choose assignment, reversal, tour partition and
        reservation priority. Shared build_tour/earliest_start/commit perform
        physical timing and collision checks. No learned CSV generation.
        """
        ...
