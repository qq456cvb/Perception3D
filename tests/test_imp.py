import unittest
import pkgutil
import importlib
import perception3d


def gen_testcase(full_name):

    def _testcase(self: unittest.TestCase):
        importlib.import_module(full_name)

    return _testcase


def onerror(name):
    print("Error importing module %s" % name)
    import traceback
    traceback.print_exc()


members = {}
for mi in pkgutil.walk_packages(perception3d.__path__, prefix=perception3d.__name__ + "."):
    disp_name = mi.name.replace("perception3d.", "").replace(".", "_")
    members["test_import_" + disp_name] = gen_testcase(mi.name)
globals()["Gen"] = type("Gen", (unittest.TestCase,), members)


if __name__ == "__main__":
    unittest.main()
