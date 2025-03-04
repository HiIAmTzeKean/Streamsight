"""
.. currentmodule:: streamsightv2.metadata

Dataset
-------------

The metadata module allows users to include metadata information corresponding to the dataset. 
The metadata classes are built on top of the :class:`Metadata`
allowing for easy extension and customization. In this module, we provide
the a few metadata that is available from public sources. The programmer is free to add more
metadatas as they see fit by defining the abstract methods that must be implemented.

It is important to note that userId and itemId in the metadata module is mapped according to
Streamsight's own internal mapping and not the original userId and itemId in the metadata.
Hence, developers should not load the metadata from source but instead implement the metadata class
and load the metadata while loading the dataset.

.. autosummary::
    :toctree: generated/

    Metadata
    MovieLens100KUserMetadata
    MovieLens100KItemMetadata
    
Example
~~~~~~~~~

The following example demonstrates how to load the metadata from the MovieLens100K dataset.

.. code-block:: python

    from streamsightv2.datasets.movielens import MovieLens100K

    dataset = MovieLens100K(fetch_metadata=True)
    data = dataset.load()

"""

from streamsightv2.metadata.base import Metadata
from streamsightv2.metadata.movielens import MovieLens100kUserMetadata, MovieLens100kItemMetadata
