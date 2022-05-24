# Create logger class with different levels of logging using different colors.

import os
import sys
import datetime as dt


class Logger(object):
    def __init__(self, filename="track.log"):
        self.log = open(filename, "a")

    def write(self, message, level=None):
        message = "[{}] {}\n".format(
            dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)

        if level == "INFO":
            print("\033[92m{}\033[0m".format(message), end="")
            self.log.write(message)
        elif level == "WARNING":
            print("\033[93m{}\033[0m".format(message), end="")
            self.log.write(message)
        elif level == "ERROR":
            print("\033[91m{}\033[0m".format(message), end="")
            self.log.write(message)
        elif level == "QUITE":
            self.log.write(message)
        else:
            print(message)
            self.log.write(message)
