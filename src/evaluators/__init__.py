import os
import pkgutil
import importlib
import inspect


def import_classes_from_package(package, package_name, prefix=""):
    package_dir = os.path.dirname(package.__file__)
    for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
        full_module_name = f"{package_name}.{module_name}"
        if module_name not in ("base", "__init__"):
            module = importlib.import_module(full_module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__:
                    key = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
                    globals()[key] = obj
            if is_pkg:
                subpackage = importlib.import_module(full_module_name)
                import_classes_from_package(
                    subpackage, full_module_name, prefix=module_name
                )


import_classes_from_package(__import__(__name__), __name__)
