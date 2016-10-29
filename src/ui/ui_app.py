import gtk
import gobject

from layer import Layer

class Sender(gobject.GObject):
    def __init__(self, signal):
        self.__gobject_init__()
        self.signal = signal

    def send(self):
        self.emit(self.signal)

gobject.type_register(Sender)


class PyApp(gtk.Window):
    def __init__(self, layers):
        super(PyApp, self).__init__()

        self.layers = layers

        self.set_title("Buttons")
        self.set_position(gtk.WIN_POS_CENTER)

        #btn1 = gtk.Button("Button")
        #btn1.set_sensitive(False)
        #btn2 = gtk.Button("Button")
        #btn3 = gtk.Button(stock=gtk.STOCK_CLOSE)
        #btn4 = gtk.Button("Button")
        #btn4.set_size_request(80, 40)

        fixed = gtk.Fixed()
        self.images = []
        curr_columns = 0
        max_rows = 0
        for layer in layers:
            image = gtk.Image()
            image.set_from_pixbuf(layer.pixbuf)
            self.images.append(image)
            fixed.put(image, curr_columns, 0)
            curr_columns += layer.columns
            max_rows = max(max_rows, layer.rows)

        self.set_size_request(curr_columns, max_rows)

        #fixed.put(btn1, 20, 30)
        #fixed.put(btn2, 100, 30)
        #fixed.put(btn3, 20, 80)
        #fixed.put(btn4, 100, 80)

        self.connect("destroy", gtk.main_quit)
        self.add(fixed)

    def register_sender(self, signal_name, callback):
        sender = Sender(signal_name)
        gobject.signal_new(signal_name, Sender,
            gobject.SIGNAL_RUN_FIRST, gobject.TYPE_NONE, ())
        sender.connect(signal_name, callback)
        return sender

    def run(self):
        self.show_all()
        gtk.main()

    def update(self, sender):
        for layer,image in zip(self.layers, self.images):
            image.set_from_pixbuf(layer.pixbuf)
        #print "user callback reacts to read_signal"

    def kill(self, sender):
        gtk.main_quit()
