"""This module contains the client class that will communicate with the server."""

import logging

import rpyc


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    conn = rpyc.connect("rpi32.local", 18861)
    print(conn.root.run_adc_sample())
    conn.close()
