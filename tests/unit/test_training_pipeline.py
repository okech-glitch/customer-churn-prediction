import os
import importlib.util

def test_training_script_exists():
    assert os.path.exists(os.path.join('scripts', 'train_models.py'))


def test_import_training_script_module():
    spec = importlib.util.spec_from_file_location('train_models', os.path.join('scripts','train_models.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, 'ChurnPredictor')
