from transforms.api import Pipeline

from myproject import datasets, target_datasets


my_pipeline = Pipeline()
my_pipeline.discover_transforms(datasets, target_datasets)
