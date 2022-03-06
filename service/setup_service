#!/bin/bash

sudo addgroup --system soundassist
sudo adduser --system --ingroup soundassist soundassist
sudo adduser soundassist video

su - soundassist

sudo cp soundassist.service /lib/systemd/system/soundassist.service
sudo chmod 755 /lib/systemd/system/soundassist.service

sudo systemctl daemon-reload
sudo systemctl start soundassist
