#!/bin/bash

sudo nmcli con add type wifi ifname wlan0 mode ap con-name WIFI_AP ssid $HOSTNAME
sudo nmcli con modify WIFI_AP 802-11-wireless.band bg
sudo nmcli con modify WIFI_AP ipv4.method shared
sudo nmcli con modify WIFI_AP ipv4.addr 192.168.5.1/24
sudo nmcli con up WIFI_AP
