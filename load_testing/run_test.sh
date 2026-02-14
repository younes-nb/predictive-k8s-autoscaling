#!/bin/bash

TARGET_URL="https://online-boutique.younesnb.linkpc.net"

echo "Starting Locust targeting $TARGET_URL"
locust -f locustfile.py --host $TARGET_URL