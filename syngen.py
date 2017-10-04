from ctypes import *

class BaseArray(Structure):
    _fields_=[("size",c_int), ("type",c_uint), ("data",c_void_p)]

class FloatArray(Structure):
    def __init__(self, base_array):
        self.size = base_array.size
        self.data = cast(base_array.data, POINTER(c_float))

class IntArray(Structure):
    def __init__(self, base_array):
        self.size = base_array.size
        self.data = cast(base_array.data, POINTER(c_int))

def build_array(base_array):
    # See POINTER_TYPE in extern.h
    (F_P, I_P, V_P) = (1,2,3)

    if base_array.type == F_P:
        return FloatArray(base_array)
    elif base_array.type == I_P:
        return IntArray(base_array)
    elif base_array.type == V_P:
        raise ValueError

_syn = CDLL('synaptogenesis.so')
_syn.get_neuron_data.restype = BaseArray
_syn.get_layer_data.restype = BaseArray
_syn.get_connection_data.restype = BaseArray
_syn.get_weight_matrix.restype = BaseArray

class CObject:
    def __init(self):
        self.obj = None

    def __del__(self):
        if _syn is not None and self.obj is not None:
            _syn.destroy(self.obj)

class Properties(CObject):
    def __init__(self, props=dict()):
        self.obj = _syn.create_properties()

        # Dictionaries of Property Objects
        self.properties = dict()
        self.children = dict()
        self.arrays = dict()

        for k,v in props.iteritems():
            if type(v) is dict:
                self.add_child(k, v)
            elif type(v) is list:
                for x in v:
                    if type(x) is dict:
                        self.add_to_array(k, x)
            else:
                self.add_property(k, v)

    def add_property(self, key, val):
        self.properties[key] = val
        _syn.add_property(self.obj, key, str(val))

    def add_child(self, key, child):
        child = Properties(child)
        self.children[key] = child
        _syn.add_child(self.obj, key, child.obj)

    def add_to_array(self, key, dictionary):
        props = Properties(dictionary)
        if key not in self.arrays:
            self.arrays[key] = []
        self.arrays[key].append(props)
        _syn.add_to_array(self.obj, key, props.obj)


class Environment(CObject):
    def __init__(self, env):

        if type(env) is str:
            self.obj = _syn.load_env(env)
        elif type(env) is dict:
            env = Properties(env)
            self.obj = _syn.create_environment(env.obj)
            del env

    def save(self, filename):
        return _syn.save_env(self.obj, filename)


class Network(CObject):
    def __init__(self, net):

        if type(net) is str:
            self.obj = _syn.load_net(net)
        elif type(net) is dict:
            net = Properties(net)
            self.obj = _syn.create_network(net.obj)
            del net
        self.state = None

    def save(self, filename):
        return _syn.save_net(self.obj, filename)

    def build_state(self):
        if self.state is not None:
            _syn.destroy(self.state)
        self.state = _syn.build_state(self.obj)

    def load_state(self, filename):
        if self.state is None: self.build_state()
        _syn.load_state(self.state, filename)

    def save_state(self, filename):
        if self.state is None: self.build_state()
        _syn.save_state(self.state, filename)

    def get_neuron_data(self, structure, layer, key):
        if self.state is None: self.build_state()
        return build_array(
            _syn.get_neuron_data(self.state, structure, layer, key))

    def get_layer_data(self, structure, layer, key):
        if self.state is None: self.build_state()
        return build_array(
            _syn.get_layer_data(self.state, structure, layer, key))

    def get_connection_data(self, structure, connection, key):
        if self.state is None: self.build_state()
        return build_array(
            _syn.get_connection_data(self.state, connection, key))

    def get_weight_matrix(self, connection):
        if self.state is None: self.build_state()
        return build_array(
            _syn.get_weight_matrix(self.state, connection))

    def run(self, environment, args):
        # Build state
        if self.state is None: self.build_state()

        # Build environment
        if not isinstance(environment, Environment):
            environment = Environment(environment)

        # Build args
        if not isinstance(args, Properties):
            args = Properties(args)

        if not bool(_syn.run(self.obj, environment.obj, self.state, args.obj)):
            print("Failed to run network!")
