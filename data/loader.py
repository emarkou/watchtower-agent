from abc import ABC, abstractmethod
from typing import Literal
import pandas as pd


class BaseDataLoader(ABC):
    @abstractmethod
    def load_reference(self) -> tuple[pd.DataFrame, pd.Series]: ...

    @abstractmethod
    def load_current_window(self) -> tuple[pd.DataFrame, pd.Series]: ...

    @abstractmethod
    def get_feature_names(self) -> list[str]: ...

    @abstractmethod
    def get_task_type(self) -> Literal["classification", "regression"]: ...
