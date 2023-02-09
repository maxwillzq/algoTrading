import unittest

from algotrading.utils.registry import Registry

class TestRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = Registry("test_registry")

    def test_default_name(self):
        @self.registry.register
        def my_func():
            pass
        self.assertIn("my_func", self.registry)

    def test_custom_name(self):
        @self.registry.register("custom_name")
        def my_func():
            pass
        self.assertIn("custom_name", self.registry)
        self.assertNotIn("my_func", self.registry)

    def test_register_func(self):
        def my_func():
            pass
        self.registry.register()(my_func)
        self.assertIn("my_func", self.registry)

    def test_value_transformer(self):
        class A:
            pass
        self.registry._value_transformer = (lambda k, v: k)
        self.registry.register()(A)
        self.assertEqual(self.registry["a"], "a")

    def test_validator(self):
        def validator(key, value):
            raise ValueError("Invalid value")
        self.registry._validator = validator
        with self.assertRaises(ValueError):
            @self.registry.register
            def my_func():
                pass

    def test_on_set(self):
        called = False
        def on_set(key, value):
            nonlocal called
            called = True
        self.registry._on_set = on_set
        @self.registry.register
        def my_func():
            pass
        self.assertTrue(called)

if __name__ == '__main__':
    unittest.main()
