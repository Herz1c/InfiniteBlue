#!/bin/sh
# Keeps the WSL VM from holding gigabytes of Linux file cache that Windows cannot use (user's RAM
# cap, 2026-10-20): every 10 minutes the page cache is dropped. Must run as root, started from
# Windows with:  wsl -d Ubuntu-24.04 -u root --exec setsid nohup sh /mnt/c/Users/honza/Desktop/InfiniteChess/tools/dropcache.sh
# Stops with WSL (no pid file; check with: pgrep -f dropcache.sh).
while true; do
  sleep 600
  sync
  echo 1 > /proc/sys/vm/drop_caches
done
