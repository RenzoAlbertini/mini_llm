import importlib
import io
import unittest


TEST_MODULES = [
    ("tokenizer", "tests.test_tokenizer"),
    ("model_forward", "tests.test_model_forward"),
    ("generate", "tests.test_generate"),
    ("end_to_end", "tests.test_end_to_end"),
]


def run_module(label, module_name):
    module = importlib.import_module(module_name)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    status = "OK" if result.wasSuccessful() else "FAIL"
    skipped = len(result.skipped)
    print(f"{label:14s} {status} | run={result.testsRun} skipped={skipped} failures={len(result.failures)} errors={len(result.errors)}")
    if not result.wasSuccessful():
        print(stream.getvalue())
    return result.wasSuccessful()


def main():
    print("mini_llm test runner")
    print("====================")
    results = []
    for label, module_name in TEST_MODULES:
        try:
            results.append(run_module(label, module_name))
        except Exception as exc:
            print(f"{label:14s} FAIL | import error: {exc}")
            results.append(False)

    if all(results):
        print("summary: OK")
        return 0
    print("summary: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
