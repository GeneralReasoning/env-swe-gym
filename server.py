from openreward.environments import Server

from swegym import SWEGym

if __name__ == "__main__":
    server = Server([SWEGym])
    server.run()