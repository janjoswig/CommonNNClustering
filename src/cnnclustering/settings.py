import pathlib
from typing import Dict
from typing import Optional, Union

import numpy as np
import yaml


class MetaSettings(type):
    """Metaclass to inherit class with class properties

    Classes constructed with this metaclass have a :attr:`__defaults`
    class attribute that can be accessed as a property :attr:`defaults`
    from the class.
    """

    __defaults: Dict[str, str] = {}

    @property
    def defaults(cls):
        return cls.__dict__[f"_{cls.__name__}__defaults"]


class Settings(dict, metaclass=MetaSettings):
    """Class to expose and handle configuration

    Inherits from :py:class:`MetaSettings` to allow access to the class
    attribute :py:attr:`__defaults` as a property :py:attr:`defaults`.

    Also derived from basic type :py:class:`dict`.

    The user can sublclass this class :py:class:`Settings` to provide e.g.
    a different set of default values as :py:attr:`__defaults`.
    """

    # Defaults shared by all instances of this class
    # Name-mangling allows different defaults in subclasses
    __defaults = {
        'default_cnn_cutoff': "1",
        'default_cnn_offset': "0",
        'default_radius_cutoff': "1",
        'default_member_cutoff': "2",
        'default_fit_policy': "conservative",
        'default_predict_policy': "conservative",
        'float_precision': 'sp',
        'int_precision': 'sp',
        }

    @property
    def defaults(self):
        """Return class attribute from instance"""
        return type(self).__defaults

    __float_precision_map = {
        'hp': np.float16,
        'sp': np.float32,
        'dp': np.float64,
    }

    __int_precision_map = {
        'qp': np.int8,
        'hp': np.int16,
        'sp': np.int32,
        'dp': np.int64,
    }

    @property
    def int_precision_map(cls):
        return cls.__int_precision_map

    @property
    def float_precision_map(cls):
        return cls.__float_precision_map

    @property
    def cfgfile(self):
        return self._cfgfile

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(type(self).__defaults)
        self.update(*args, **kwargs)

        self._cfgfile = None

    def __setitem__(self, key, val):
        if key in type(self).__defaults:
            super().__setitem__(key, val)
        else:
            print(f"Unknown option: {key}")

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def configure(
            self, path: Optional[Union[pathlib.Path, str]] = None,
            reset: bool = False):
        """Configuration file reading

        Reads a yaml configuration file ``.corerc`` from a given path or
        one of the standard locations in the following order of
        priority:

            - current working directory
            - user home directory

        Args:
            path: Path to a configuration file.
            reset: Reset to defaults. If True, `path` is ignored
               and no configuration file is read.
        """

        if reset:
            self.update(type(self).__defaults)
        else:
            if path is None:
                path = []
            else:
                path = [path]

            path.extend([pathlib.Path.cwd() / ".cnnclusteringrc",
                         pathlib.Path.home() / ".cnnclusteringrc"])

            places = iter(path)

            # find configuration file
            while True:
                try:
                    cfgfile = next(places)
                except StopIteration:
                    self._cfgfile = None
                    break
                else:
                    if cfgfile.is_file():
                        with open(cfgfile, 'r') as ymlfile:
                            self.update(yaml.load(
                                ymlfile, Loader=yaml.SafeLoader
                                ))

                        self._cfgfile = cfgfile
                        break
