from streamsight.datasets import AmazonMusicDataset
from streamsight.matrix import InteractionMatrix

dataset = AmazonMusicDataset()
im = dataset.load()
assert type(im) == InteractionMatrix
