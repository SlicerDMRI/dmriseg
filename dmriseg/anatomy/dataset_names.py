# -*- coding: utf-8 -*-

import enum


class DatasetShortName(enum.Enum):
    HCP = "hcp"
    HCP_YA = "hcp-ya"


class DatasetLongName(enum.Enum):
    HCP = "Human Connectome Project"
    HCP_YA = "Human Connectome Project - Young Adult"
