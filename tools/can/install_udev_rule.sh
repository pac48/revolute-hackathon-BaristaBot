#!/usr/bin/env bash
# Installs 80-rebot-can.rules (needs sudo) and applies it to the adapters that are plugged in now.
#   tools/can/install_udev_rule.sh            # on the PC with the CAN adapters
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo install -m 0644 "$here/80-rebot-can.rules" /etc/udev/rules.d/80-rebot-can.rules
sudo udevadm control --reload-rules
# Apply it to the adapters already plugged in: replay their "add" events (an adapter that is already up
# keeps its settings: the bitrate can only be set while down, and that step just fails).
sudo udevadm trigger --subsystem-match=net --sysname-match='can*' --action=add
sleep 1
ip -details -br link show type can
