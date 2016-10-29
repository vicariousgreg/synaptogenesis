import threading
import sys

import gtk
import gobject

from streamer import Streamer
from layer import Layer
from ui_app import PyApp, Sender

if __name__ == "__main__":
    fifo_name = sys.argv[1]
    out_type = sys.argv[2]
    streamer = Streamer(fifo_name, out_type)

    # Read layer information from FIFO
    layers = streamer.read_layers()

    # Create app
    gobject.threads_init()
    app = PyApp(layers)
    read_sender = app.register_sender("read", app.update)
    kill_sender = app.register_sender("kill", app.kill)

    # Launch helper and UI
    thread = threading.Thread(target=Streamer.read_loop, args=(streamer, layers, read_sender, kill_sender))
    thread.daemon = True
    thread.start()
    app.run()


    print("Exiting!")
