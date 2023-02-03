# coding=utf-8
"""Tests for registry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from algotrading.utils import registry

# pylint: disable=unused-variable,unused-argument


class RegistryClassTest(absltest.TestCase):
    """Test of base registry.Registry class."""

    def testGetterSetter(self):
        r = registry.Registry("test_registry")
        r["hello"] = lambda: "world"
        r["a"] = lambda: "b"
        self.assertEqual(r["hello"](), "world")
        self.assertEqual(r["a"](), "b")

    def testDefaultKeyFn(self):
        r = registry.Registry("test", default_key_fn=lambda x: x().upper())
        r.register()(lambda: "hello")
        self.assertEqual(r["HELLO"](), "hello")

    def testNoKeyProvided(self):
        r = registry.Registry("test")

        def f():
            return 3

        r.register(f)
        self.assertEqual(r["f"](), 3)

    def testMembership(self):
        r = registry.Registry("test_registry")
        r["a"] = lambda: None
        r["b"] = lambda: 4
        self.assertTrue("a" in r)
        self.assertTrue("b" in r)

    def testIteration(self):
        r = registry.Registry("test_registry")
        r["a"] = lambda: None
        r["b"] = lambda: 4
        self.assertEqual(sorted(r), ["a", "b"])

    def testLen(self):
        r = registry.Registry("test_registry")
        self.assertEqual(len(r), 0)
        r["a"] = lambda: None
        self.assertEqual(len(r), 1)
        r["b"] = lambda: 4
        self.assertEqual(len(r), 2)

    def testTransformer(self):
        r = registry.Registry("test_registry", value_transformer=lambda x, y: x + y())
        r.register(3)(lambda: 5)
        r.register(10)(lambda: 12)
        self.assertEqual(r[3], 8)
        self.assertEqual(r[10], 22)
        self.assertEqual(set(r.values()), set((8, 22)))
        self.assertEqual(set(r.items()), set(((3, 8), (10, 22))))

    def testGet(self):
        r = registry.Registry("test_registry", value_transformer=lambda k, v: v())
        r["a"] = lambda: "xyz"
        self.assertEqual(r.get("a"), "xyz")
        self.assertEqual(r.get("a", 3), "xyz")
        self.assertIsNone(r.get("b"))
        self.assertEqual(r.get("b", 3), 3)


if __name__ == "__main__":
    absltest.main()
