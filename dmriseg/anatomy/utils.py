# -*- coding: utf-8 -*-

import enum


class Axis(enum.Enum):
    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"

    @classmethod
    def _missing_(cls, value):
        choices = list(cls.__members__)
        raise ValueError(
            f"Unsupported value:\n" f"Found: {value}; Available: {choices}"
        )
