#!/bin/bash

set -e

DEST="$HOME/Documents/self_organization_percolation/SOP_data/tests_data/"
SRC="Documents/self_organization_percolation/SOP_data/tests_data/"

SOCKET_GUI="/tmp/ssh-gui-$USER"
SOCKET_LAB="/tmp/ssh-lab-$USER"

echo "=== Autenticação GUI ==="
ssh -fN \
    -o ControlMaster=yes \
    -o ControlPath="$SOCKET_GUI" \
    gui

echo
echo "=== Autenticação LAB ==="
ssh -fN \
    -o ControlMaster=yes \
    -o ControlPath="$SOCKET_LAB" \
    lab

echo
echo "=== Iniciando transferências simultâneas ==="

rsync -av --ignore-existing \
    -e "ssh -o ControlPath=$SOCKET_GUI" \
    "gui:$SRC" \
    "$DEST" &
PID_GUI=$!

rsync -av --ignore-existing \
    -e "ssh -o ControlPath=$SOCKET_LAB" \
    "lab:$SRC" \
    "$DEST" &
PID_LAB=$!

wait "$PID_GUI"
STATUS_GUI=$?

wait "$PID_LAB"
STATUS_LAB=$?

echo
echo "=== Transferências finalizadas ==="
echo "GUI: $STATUS_GUI"
echo "LAB: $STATUS_LAB"

ssh -O exit -o ControlPath="$SOCKET_GUI" gui 2>/dev/null || true
ssh -O exit -o ControlPath="$SOCKET_LAB" lab 2>/dev/null || true
