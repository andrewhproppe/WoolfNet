"""
Config utility functions (loading, saving, and formatting configuration files
and objects)
"""

from typing import Any, Dict

import yaml
from cloudpathlib import CloudPath


class InlineListDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(InlineListDumper, self).increase_indent(flow=flow, indentless=indentless)


def inline_list_representer(dumper, value):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)


yaml.add_representer(list, inline_list_representer)


class DotDict:
    """
    A dictionary subclass that allows dot notation access to nested dictionaries
    loaded from YAML files.

    Parameters
    ----------
    dictionary : Dict[str, Any]
        The dictionary to convert to a dot-notation accessible object.
    """

    def __init__(self, dictionary: Dict[str, Any]):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)

    def keys(self):
        """
        Return dictionary keys
        """
        return self.__dict__.keys()

    def values(self):
        """
        Return dictionary values
        """
        return self.__dict__.values()

    def items(self):
        """
        Return dictionary items
        """
        return self.__dict__.items()

    def __getattr__(self, key: str) -> Any:
        """
        Custom attribute access method for handling missing keys.

        Parameters
        ----------
        key : str
            The attribute name being accessed.
        """
        raise AttributeError(f"Config has no attribute '{key}'")

    def __str__(self) -> str:
        """
        String representation of the DotDict.
        """
        return str(self.__dict__)

    def __repr__(self) -> str:
        """
        Official string representation of the DotDict.
        """
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the instance of a DotDict to a standard
        dictionary
        """

        output = {}
        for key, value in self.items():
            if isinstance(value, DotDict):
                output[key] = value.to_dict()
            else:
                output[key] = value

        return output

    def add(self, other: "DotDict") -> "DotDict":
        """
        Combine two DotDict instances into a new DotDict.

        Parameters
        ----------
        other : DotDict
            Another DotDict to combine with this one.
        """
        combined = self.to_dict()
        other_dict = other.to_dict()
        combined.update(other_dict)
        return DotDict(combined)


def load_yaml(file_path: str) -> DotDict:
    """
    *AI Generated*

    Load YAML file and return a DotDict object for dot notation access.

    Parameters
    ----------
    file_path : str
        Path to the YAML file to be loaded.

    Returns
    -------
    DotDict
        Dictionary-like object that allows dot notation access to nested values.
    """
    if str(file_path).startswith("s3:/"):
        if not str(file_path).startswith("s3://"):
            # Fix single-slash case (pathlib strips the second slash)
            file_path = str(file_path).replace("s3:/", "s3://")
        file_path = CloudPath(file_path)
    with open(file_path, "r") as file:
        yaml_content = yaml.safe_load(file)
        return DotDict(yaml_content)


def save_yaml(file_path: str, data: Dict):
    """
    Save a dict to a .yml, either locally or on S3.
    """
    if str(file_path).startswith("s3:/"):
        if not str(file_path).startswith("s3://"):
            # Fix single-slash case (pathlib strips the second slash)
            file_path = str(file_path).replace("s3:/", "s3://")
        file_path = CloudPath(file_path)

    with file_path.open("w", encoding="utf-8") as fh:
        yaml.dump(
            data,
            fh,
            default_flow_style=False,
            sort_keys=False,
            indent=3,
            Dumper=InlineListDumper,
        )
