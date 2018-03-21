#!/usr/bin/env python

import re
import os
import glob
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from allennlp.commands.train import train_model_from_file, train_model
from allennlp.common import Params

def train_fixture(config_prefix: str) -> None:
    config_file = config_prefix + 'experiment.json'
    serialization_dir = config_prefix + 'serialization'

    # Train model doesn't like it if we have incomplete serialization
    # directories, so remove them if they exist.
    if os.path.exists(serialization_dir):
        shutil.rmtree(serialization_dir)

    # train the model
    train_model_from_file(config_file, serialization_dir)

    # remove unnecessary files
    shutil.rmtree(os.path.join(serialization_dir, "log"))

    for filename in glob.glob(os.path.join(serialization_dir, "*")):
        if filename.endswith(".log") or filename.endswith(".json") or re.search(r"epoch_[0-9]+\.th$", filename):
            os.remove(filename)

def train_fixture_gpu(config_prefix: str) -> None:
    config_file = config_prefix + 'experiment.json'
    serialization_dir = config_prefix + 'serialization'
    params = Params.from_file(config_file)
    params["trainer"]["cuda_device"] = 0

    # train this one to a tempdir
    tempdir = tempfile.gettempdir()
    train_model(params, tempdir)

    # now copy back the weights and and archived model
    shutil.copy(os.path.join(tempdir, "best.th"), os.path.join(serialization_dir, "best_gpu.th"))
    shutil.copy(os.path.join(tempdir, "model.tar.gz"), os.path.join(serialization_dir, "model_gpu.tar.gz"))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "gpu":
        train_fixture_gpu("tests/fixtures/srl/")
    else:
<<<<<<< HEAD
        train_fixture("tests/fixtures/decomposable_attention/experiment.json", "tests/fixtures/decomposable_attention/serialization")
        train_fixture("tests/fixtures/bidaf/experiment.json", "tests/fixtures/bidaf/serialization")
        train_fixture("tests/fixtures/srl/experiment.json", "tests/fixtures/srl/serialization")
        train_fixture("tests/fixtures/coref/experiment.json", "tests/fixtures/coref/serialization")
        train_fixture("tests/fixtures/constituency_parser/experiment_no_evalb.json", "tests/fixtures/constituency_parser/serialization")
        train_fixture("tests/fixtures/encoder_decoder/wikitables_semantic_parser/experiment.json",
                      "tests/fixtures/encoder_decoder/wikitables_semantic_parser/serialization")
=======
        models = [
                'bidaf',
                'constituency_parser',
                'coref',
                'decomposable_attention',
                'encoder_decoder/simple_seq2seq',
                'srl',
                ]
        for model in models:
            train_fixture(f"tests/fixtures/{model}/")
>>>>>>> 760853c5f9a3b3d470bbdcc65526f2fac012514a
