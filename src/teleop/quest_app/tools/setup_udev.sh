#!/usr/bin/env bash
# Lets adb talk to Meta Quest headsets (USB vendor 0x2833) without root.
# Run once per machine:  sudo src/teleop/quest_app/tools/setup_udev.sh
# Then unplug/replug the headset and accept "Allow USB debugging" inside it.
set -euo pipefail

RULE_FILE=/etc/udev/rules.d/51-android-quest.rules
RULE='SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0666", GROUP="plugdev", TAG+="uaccess"'

if [[ $EUID -ne 0 ]]; then
    echo "needs root: sudo $0" >&2
    exit 1
fi

echo "$RULE" > "$RULE_FILE"
chmod 644 "$RULE_FILE"
udevadm control --reload-rules
udevadm trigger --subsystem-match=usb --attr-match=idVendor=2833
echo "wrote $RULE_FILE:"
cat "$RULE_FILE"
echo "Now unplug and replug the headset, accept USB debugging in it, then run: adb devices"
