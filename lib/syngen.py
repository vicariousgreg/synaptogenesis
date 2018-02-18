from ctypes import *
from collections import OrderedDict
from json import dumps

class CArray(Structure):
    _fields_=[("size",c_int),
              ("type",c_uint),
              ("data",c_void_p),
              ("owner",c_bool)]

class PArray:
    def __init__(self, size, ptr):
        self.size = size
        self.data = ptr

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        curr = 0
        while (curr < self.size):
            yield self.data[curr]
            curr += 1

class FloatArray(PArray):
    def __init__(self, size, ptr):
        self.size = size
        self.data = cast(ptr, POINTER(c_float))

    def to_list(self):
        return [float(x) for x in self]

class IntArray(PArray):
    def __init__(self, size, ptr):
        self.size = size
        self.data = cast(ptr, POINTER(c_int))

    def to_list(self):
        return [int(x) for x in self]

class StringArray(PArray):
    def __init__(self, size, ptr):
        self.size = size
        self.data = cast(ptr, POINTER(POINTER(c_char)))

class VoidArray(PArray):
    def __init__(self, size, ptr):
        self.size = size
        self.data = cast(ptr, POINTER(c_void_p))

def build_array(base_array):
    # See POINTER_TYPE in extern.h
    (F_P, I_P, S_P, P_P, V_P) = (1,2,3,4,5)

    if base_array.type == F_P:
        return FloatArray(base_array.size, base_array.data)
    elif base_array.type == I_P:
        return IntArray(base_array.size, base_array.data)
    elif base_array.type == S_P:
        return StringArray(base_array.size, base_array.data)
    elif base_array.type == P_P:
        return VoidArray(base_array.size, base_array.data)
    elif base_array.type == V_P:
        return VoidArray(base_array.size, base_array.data)

_syn = CDLL('libsyngen.so')

_syn.free_array.argtypes = (CArray, )
_syn.free_array_deep.argtypes = (CArray, )

_syn.create_properties.restype = c_void_p
_syn.add_property.argtypes = (c_void_p, c_char_p, c_char_p)
_syn.add_child.argtypes = (c_void_p, c_char_p, c_void_p)
_syn.add_to_array.argtypes = (c_void_p, c_char_p, c_char_p)
_syn.add_to_child_array.argtypes = (c_void_p, c_char_p, c_void_p)

_syn.get_keys.argtypes = (c_void_p,)
_syn.get_keys.restype = CArray
_syn.get_child_keys.argtypes = (c_void_p,)
_syn.get_child_keys.restype = CArray
_syn.get_array_keys.argtypes = (c_void_p,)
_syn.get_array_keys.restype = CArray
_syn.get_child_array_keys.argtypes = (c_void_p,)
_syn.get_child_array_keys.restype = CArray

_syn.get_property.restype = c_char_p
_syn.get_property.argtypes = (c_void_p, c_char_p)
_syn.get_child.restype = c_void_p
_syn.get_child.argtypes = (c_void_p, c_char_p)
_syn.get_array.restype = CArray
_syn.get_array.argtypes = (c_void_p, c_char_p)
_syn.get_child_array.restype = CArray
_syn.get_child_array.argtypes = (c_void_p, c_char_p)

_syn.create_environment.restype = c_void_p
_syn.create_environment.argtypes = (c_void_p,)
_syn.load_env.restype = c_void_p
_syn.load_env.argtypes = (c_char_p,)
_syn.save_env.restype = c_bool
_syn.save_env.argtypes = (c_void_p, c_char_p)

_syn.create_network.restype = c_void_p
_syn.create_network.argtypes = (c_void_p,)
_syn.load_net.restype = c_void_p
_syn.load_net.argtypes = (c_char_p,)
_syn.save_net.restype = c_bool
_syn.save_net.argtypes = (c_void_p, c_char_p)

_syn.build_state.restype = c_void_p
_syn.build_state.argtypes = (c_void_p,)
_syn.load_state.restype = c_bool
_syn.load_state.argtypes = (c_void_p, c_char_p)
_syn.save_state.restype = c_bool
_syn.save_state.argtypes = (c_void_p, c_char_p)

_syn.get_neuron_data.restype = CArray
_syn.get_neuron_data.argtypes = (c_void_p, c_char_p, c_char_p, c_char_p)
_syn.get_layer_data.restype = CArray
_syn.get_layer_data.argtypes = (c_void_p, c_char_p, c_char_p, c_char_p)
_syn.get_connection_data.restype = CArray
_syn.get_connection_data.argtypes = (c_void_p, c_char_p, c_char_p)
_syn.get_weight_matrix.restype = CArray
_syn.get_weight_matrix.argtypes = (c_void_p, c_char_p, c_char_p)

_syn.run.restype = c_void_p
_syn.run.argtypes = (c_void_p, c_void_p, c_void_p, c_void_p)

_syn.destroy.argtypes = (c_void_p,)

_syn.get_cpu.restype = c_int
_syn.get_gpus.restype = CArray
_syn.get_all_devices.restype = CArray

_syn.set_suppress_output.argtypes = (c_bool,)
_syn.set_warnings.argtypes = (c_bool,)
_syn.set_debug.argtypes = (c_bool,)

def get_cpu():
    return _syn.get_cpu()

def get_gpus():
    gpus_object = _syn.get_gpus()
    gpus = build_array(gpus_object).to_list()
    _syn.free_array(gpus_object)
    return gpus

def get_all_devices():
    devices_object = _syn.get_all_devices()
    devices = build_array(devices_object).to_list()
    _syn.free_array(devices_object)
    return devices

def set_suppress_output(val):
    _syn.set_suppress_output(val)

def set_warnings(val):
    _syn.set_warnings(val)

def set_debug(val):
    _syn.set_debug(val)

def interrupt_engine():
    _syn.interrupt_engine()

class CObject:
    def __init(self):
        self.obj = None

    def __del__(self):
        if _syn is not None and self.obj is not None:
            _syn.destroy(self.obj)

class Properties(CObject):
    def __init__(self, props=dict()):
        # Dictionaries of Property Objects
        self.properties = OrderedDict()
        self.children = OrderedDict()
        self.arrays = OrderedDict()
        self.child_arrays = OrderedDict()

        # If dictionary, build C properties
        if type(props) is dict:
            self.obj = _syn.create_properties()
            for k,v in props.iteritems():
                if type(v) is dict:
                    self.add_child(k, v)
                elif type(v) is list:
                    for x in v:
                        if type(x) is dict:
                            self.add_to_child_array(k, x)
                        else:
                            self.add_to_array(k, str(x))
                else:
                    self.add_property(k, v)
        # If int, interpret as C properties pointer and retrieve data
        elif type(props) is int:
            self.obj = props

            # Get string properties
            keys_obj = _syn.get_keys(props)
            keys = build_array(keys_obj)
            for i in range(keys.size):
                key = keys.data[i]
                key_str = string_at(cast(key, c_char_p))
                self.properties[key_str] = string_at(
                    _syn.get_property(props, key))

            _syn.free_array_deep(keys_obj)

            # Get child properties
            keys_obj = _syn.get_child_keys(props)
            keys = build_array(keys_obj)
            for i in range(keys.size):
                key = keys.data[i]
                key_str = string_at(cast(key, c_char_p))
                self.children[key_str] = Properties(
                    _syn.get_child(props, key))
            _syn.free_array_deep(keys_obj)

            # Get array properties
            keys_obj = _syn.get_array_keys(props)
            keys = build_array(keys_obj)
            for i in range(keys.size):
                key = keys.data[i]
                key_str = string_at(cast(key, c_char_p))
                arr_obj = _syn.get_array(props, key)
                arr = build_array(arr_obj)

                if key_str not in self.arrays:
                    self.arrays[key_str] = []

                for j in range(arr.size):
                    self.arrays[key_str].append(
                        string_at(cast(arr.data[j], c_char_p)))
                _syn.free_array(arr_obj)
            _syn.free_array_deep(keys_obj)

            # Get child array properties
            keys_obj = _syn.get_child_array_keys(props)
            keys = build_array(keys_obj)
            for i in range(keys.size):
                key = keys.data[i]
                key_str = string_at(cast(key, c_char_p))
                arr_obj = _syn.get_child_array(props, key)
                arr = build_array(arr_obj)

                if key_str not in self.child_arrays:
                    self.child_arrays[key_str] = []

                for j in range(arr.size):
                    self.child_arrays[key_str].append(
                        Properties(arr.data[j]))
                _syn.free_array(arr_obj)
            _syn.free_array_deep(keys_obj)

    def to_dict(self):
        out = OrderedDict()
        for k,v in self.properties.iteritems():
            out[k] = v
        for k,v in self.children.iteritems():
            out[k] = v.to_dict()
        for k,v in self.arrays.iteritems():
            out[k] = [str(p) for p in v]
        for k,v in self.child_arrays.iteritems():
            out[k] = [p.to_dict() for p in v]
        return out

    def __str__(self):
        return dumps(self.to_dict(), indent=4)

    def add_property(self, key, val):
        self.properties[key] = val
        _syn.add_property(self.obj, key, str(val))

    def add_child(self, key, child):
        child = Properties(child)
        self.children[key] = child
        _syn.add_child(self.obj, key, child.obj)

    def add_to_array(self, key, val):
        if key not in self.arrays:
            self.arrays[key] = []
        self.arrays[key].append(val)
        _syn.add_to_array(self.obj, key, val)

    def add_to_child_array(self, key, dictionary):
        props = Properties(dictionary)
        if key not in self.child_arrays:
            self.child_arrays[key] = []
        self.child_arrays[key].append(props)
        _syn.add_to_child_array(self.obj, key, props.obj)


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

    def get_weight_matrix(self, connection, key="weights"):
        if self.state is None: self.build_state()
        return build_array(
            _syn.get_weight_matrix(self.state, connection, key))

    def run(self, environment, args=dict()):
        # Build state
        if self.state is None: self.build_state()

        # Build environment
        if not isinstance(environment, Environment):
            environment = Environment(environment)

        # Build args
        if not isinstance(args, Properties):
            args = Properties(args)

        report = _syn.run(self.obj, environment.obj, self.state, args.obj)
        if report is None: print("Failed to run network!")
        return Properties(report)


def create_io_callback(f):
    cb = CFUNCTYPE(None, c_int, c_int, c_void_p)(f)
    return (cb, cast(cb, c_void_p).value)

def create_distance_weight_callback(f):
    cb = CFUNCTYPE(None, c_int, c_int, c_void_p, c_void_p)(f)
    return (cb, cast(cb, c_void_p).value)
