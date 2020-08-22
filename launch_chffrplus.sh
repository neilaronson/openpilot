#!/usr/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -z "$BASEDIR" ]; then
  BASEDIR="/data/openpilot"
fi

if [ -z "$PASSIVE" ]; then
  export PASSIVE="1"
fi

STAGING_ROOT="/data/safe_staging"

function launch {
  # Wifi scan
  wpa_cli IFNAME=wlan0 SCAN

  # no cpu rationing for now
  echo 0-3 > /dev/cpuset/background/cpus
  echo 0-3 > /dev/cpuset/system-background/cpus
  echo 0-3 > /dev/cpuset/foreground/boost/cpus
  echo 0-3 > /dev/cpuset/foreground/cpus
  echo 0-3 > /dev/cpuset/android/cpus

  # change interrupt affinity
  echo 3 > /proc/irq/6/smp_affinity_list # MDSS
  echo 1 > /proc/irq/78/smp_affinity_list # Modem, can potentially lock up
  echo 2 > /proc/irq/733/smp_affinity_list # USB
  echo 2 > /proc/irq/736/smp_affinity_list # USB

  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

  # Remove old NEOS update file
  # TODO: move this code to the updater
  if [ -d /data/neoupdate ]; then
    rm -rf /data/neoupdate
  fi

  # Check for NEOS update
  if [ $(< /VERSION) != "14" ]; then
    if [ -f "$DIR/scripts/continue.sh" ]; then
      cp "$DIR/scripts/continue.sh" "/data/data/com.termux/files/continue.sh"
    fi

    "$DIR/installer/updater/updater" "file://$DIR/installer/updater/update.json"
  fi


  # handle pythonpath
  ln -sfn $(pwd) /data/pythonpath
  export PYTHONPATH="$PWD"

  # start manager
  cd selfdrive
  ./manager.py

  # if broken, keep on screen error
  while true; do sleep 1; done
}

launch
