from importlib import import_module


_EXPORTS = {
    "training.alternating": (
        "MemoryTrainingFlow",
        "PendingMemoryEdit",
        "QuestionCandidate",
    ),
    "training.dataset_builder": (
        "AttackerDatasetBuilder",
        "DatasetFiles",
        "RouteProposalBuilder",
        "RouteSelectorDatasetBuilder",
        "memory_builder_records",
        "write_verl_dataset",
    ),
}
_MODULES = {
    name: module
    for module, names in _EXPORTS.items()
    for name in names
}
__all__ = list(_MODULES)


def __getattr__(name: str):
    if module := _MODULES.get(name):
        value = getattr(import_module(module), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
