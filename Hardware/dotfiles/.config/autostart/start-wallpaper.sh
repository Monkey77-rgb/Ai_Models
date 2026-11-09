#!/bin/bash
pkill xwinwrap
pkill mplayer
sleep 1
xwinwrap -fs -fdt -ni -b -nf -ov -argb -- mplayer -wid WID -loop 0 -quiet -nosound -vo x11 "${WALLPAPER_VIDEO:-$HOME/Pictures/wallpaper.mp4}" &
