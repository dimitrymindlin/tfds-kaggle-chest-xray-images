"""kaggleChestXrayImages dataset."""

import tensorflow_datasets as tfds
import os
from pathlib import Path

_DESCRIPTION = """
Context
Pneumonia is an infection that inflames the air sacs in one or both lungs.
It kills more children younger than 5 years old each year than any other infectious disease, such as HIV infection,
malaria, or tuberculosis. Diagnosis is often based on symptoms and physical examination. Chest X-rays may help
confirm the diagnosis.

Content
This dataset contains 5,856 validated Chest X-Ray images. The images are split into a training set and a testing
set of independent patients. Images are labeled as 
(disease:NORMAL/BACTERIA/VIRUS)-(randomized patient ID)-(image number of a patient).
For details of the data collection and description, see the referenced paper below.

According to the paper, the images (anterior-posterior) were selected from retrospective cohorts of
pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou.

A previous version (v2) of this dataset is available here:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. Note that the files names are irregular in v2,
but they are fixed in the new version (v3).

Inspiration
This data will be useful for developing/training/testing classification models with convolutional neural networks.

Acknowledgements
This dataset is taken from https://data.mendeley.com/datasets/rscbjbr9sj/3.

Licence: CC BY 4.0

Reference: https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
"""

_CITATION = """
 @misc{
 kermany_2018, 
 title={Large dataset of labeled Optical Coherence Tomography (OCT) and chest X-ray images}, 
 url={https://data.mendeley.com/datasets/rscbjbr9sj/3}, 
 journal={Mendeley Data}, 
 publisher={Mendeley Data}, 
 author={Daniel Kermany, Kang Zhang, Michael Goldbaum}, 
 year={2018}, 
 month={Jun}}
 """

_Classes = [
    "NORMAL",
    "BACTERIA",
    "VIRUS",
]


class Kagglechestxrayimages(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for kaggleChestXrayImages dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'name': tfds.features.Text(),  # patient id
                'image': tfds.features.Image(shape=(224, 224, 3)),
                'image_num': tfds.features.Text(),  # image number of a patient
                'label': tfds.features.ClassLabel(names=['NORMAL', 'DISEASE']),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='https://www.kaggle.com/tolgadincer/labeled-chest-xray-images',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        #path = dl_manager.download_and_extract('https://www.kaggle.com/tolgadincer/labeled-chest-xray-images/download')
        path = dl_manager.download_kaggle_data(competition_or_dataset='tolgadincer/labeled-chest-xray-images')
        #extracted_path = dl_manager.extract(path)

        return {
            'train': self._generate_examples(os.path.join(path, 'chest_xray/train')),
            'test': self._generate_examples(os.path.join(path, 'chest_xray/test'))
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for img_path in (Path(os.path.join(path, 'NORMAL'))).glob('*.jpeg'):
            yield img_path.name, {
                'name': img_path.name.split("-")[1], # patient id
                'image': img_path,
                'label': 'NORMAL',
                'image_num': img_path.name.split("-")[2],  # image number of a patient
            }

        for img_path in (Path(os.path.join(path, 'PNEUMONIA'))).glob('*.jpeg'):
            yield img_path.name, {
                'name': img_path.name.split("-")[1], # patient id
                'image': img_path,
                'label': 'DISEASE',
                'image_num': img_path.name.split("-")[2],  # image number of a patient
            }
