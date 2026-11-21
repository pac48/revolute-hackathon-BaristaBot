"""clock_probe: time conversions and the ping stamp encoding round-trip."""

import unittest

from src.camera_drivers import clock_probe


class ClockProbeTest(unittest.TestCase):
    def test_time_round_trip(self):
        ns = 1_794_711_234_000_000_123
        t = clock_probe.to_time(ns)
        self.assertEqual((t.sec, t.nanosec), (1_794_711_234, 123))
        self.assertEqual(clock_probe.to_ns(t), ns)

    def test_ping_stamp_round_trip(self):
        t = clock_probe.to_time(1_794_711_234_000_000_123)
        self.assertEqual(clock_probe.encode_ping_stamp(t), "1794711234.000000123")
        self.assertEqual(clock_probe.decode_ping_stamp(clock_probe.encode_ping_stamp(t)), clock_probe.to_ns(t))


if __name__ == "__main__":
    unittest.main()
