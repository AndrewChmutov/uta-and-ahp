import importlib
from pathlib import Path

[
    importlib.import_module(f".{x.stem}", __package__)
    for x in Path(__file__).parent.iterdir()
    if x.suffix == ".py"
    if x.stem != "__init__"
]
