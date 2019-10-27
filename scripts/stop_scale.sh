#!/bin/bash
./pd-ctl -u http://10.9.45.61:2379 sche add reject-region
./pd-ctl -u http://10.9.45.61:2379 sche remove auto-scale-region-scheduler
