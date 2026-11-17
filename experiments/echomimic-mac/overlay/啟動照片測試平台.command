#!/bin/bash
cd "$(dirname "$0")"
open 'http://127.0.0.1:7881'
bash mac/start_test_ui.sh
read -r -p '平台已停止，按 Enter 關閉。'
