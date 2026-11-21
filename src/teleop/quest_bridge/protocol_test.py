import json
import struct
import unittest

from src.teleop.quest_bridge import protocol

# As the headset's encoder writes it (src/teleop/quest_app/core/protocol.cpp).
SAMPLE = (
    '{"type":"controllers","seq":42,"t":1.500000,'
    '"head":{"p":[0.10000,0.20000,1.60000],"q":[0.000000,0.000000,0.000000,1.000000]},'
    '"left":{"active":false,"trigger":0.0000,"grip":0.0000,"stick":[0.0000,0.0000],'
    '"buttons":{"x":0,"y":0,"stick":0,"menu":1}},'
    '"right":{"active":true,"p":[0.40000,-0.20000,0.30000],"q":[0.000000,0.000000,0.707107,0.707107],'
    '"trigger":0.7500,"grip":1.0000,"stick":[0.1000,-0.5000],"buttons":{"a":1,"b":0,"stick":0}}}'
)


class ProtocolTest(unittest.TestCase):
    def test_controllers(self):
        c = protocol.controllers_from(protocol.parse(SAMPLE))
        self.assertEqual(c.seq, 42)
        self.assertAlmostEqual(c.t, 1.5)
        self.assertEqual(c.head.p, (0.1, 0.2, 1.6))
        self.assertFalse(c.left.active)
        self.assertIsNone(c.left.pose)
        self.assertEqual(c.left.buttons, {"x": False, "y": False, "stick": False, "menu": True})
        self.assertTrue(c.right.active)
        self.assertEqual(c.right.pose.p, (0.4, -0.2, 0.3))
        self.assertAlmostEqual(c.right.pose.q[2], 0.7071068, places=5)
        self.assertEqual(c.right.trigger, 0.75)
        self.assertEqual(c.right.stick, (0.1, -0.5))
        self.assertEqual(c.right.buttons, {"a": True, "b": False, "stick": False})

    def test_missing_head_and_hands(self):
        c = protocol.controllers_from(protocol.parse('{"type":"controllers","seq":1,"t":0}'))
        self.assertIsNone(c.head)
        self.assertFalse(c.left.active)
        self.assertEqual(c.right.buttons, {"a": False, "b": False, "stick": False})

    def test_quaternion_normalized(self):
        c = protocol.controllers_from(
            protocol.parse('{"type":"controllers","head":{"p":[0,0,0],"q":[0,0,0,2]}}')
        )
        self.assertEqual(c.head.q, (0.0, 0.0, 0.0, 1.0))

    def test_errors(self):
        for bad in ["not json", "[1,2]", '{"no_type":1}', '{"type":3}']:
            with self.assertRaises(protocol.ProtocolError, msg=bad):
                protocol.parse(bad)
        for bad in [
            '{"type":"controllers","head":{"p":[0,0],"q":[0,0,0,1]}}',
            '{"type":"controllers","head":{"p":[0,0,0],"q":[0,0,0,0]}}',
            '{"type":"controllers","right":{"active":true,"p":[0,0,0]}}',
            '{"type":"controllers","right":{"active":true,"p":["a",0,0],"q":[0,0,0,1]}}',
        ]:
            with self.assertRaises(protocol.ProtocolError, msg=bad):
                protocol.controllers_from(protocol.parse(bad))

    def test_view(self):
        data = b"VIEW" + struct.pack("<HHd", 720, 792, 3.25) + b"\xff\xd8rest"
        v = protocol.parse_view(data)
        self.assertEqual((v.width, v.height, v.t, v.jpeg), (720, 792, 3.25, b"\xff\xd8rest"))
        for bad in [b"VIEW", b"XXXX" + data[4:], data[:16] + b"notjpeg"]:
            with self.assertRaises(protocol.ProtocolError):
                protocol.parse_view(bad)

    def test_gripper_pose(self):
        text = protocol.encode_gripper_pose(protocol.Pose((0.3, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)), 12.5)
        msg = json.loads(text)
        self.assertEqual(msg, {"type": "gripper_pose", "t": 12.5, "p": [0.3, 0.0, 0.2], "q": [0.0, 0.0, 0.0, 1.0]})


if __name__ == "__main__":
    unittest.main()
