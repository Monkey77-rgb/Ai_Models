#!/bin/bash
XAUTHORITY=$(cat /proc/$(pgrep -x lxqt-session)/environ 2>/dev/null | tr '\0' '\n' | grep XAUTHORITY | cut -d= -f2)
DISPLAY=:0 XAUTHORITY="$XAUTHORITY" kitty python3 "$(dirname "$(readlink -f "$0")")/agent.py"
