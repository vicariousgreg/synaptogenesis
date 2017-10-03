import ctypes

_syn = ctypes.CDLL('synaptogenesis.so')

class CObject:
    def __init(self):
        self.obj = None

    def __del__(self):
        global _syn
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
        global _syn
        self.properties[key] = val
        _syn.add_property(self.obj, key, str(val))

    def add_child(self, key, child):
        global _syn
        child = Properties(child)
        self.children[key] = child
        _syn.add_child(self.obj, key, child.obj)

    def add_to_array(self, key, dictionary):
        global _syn
        props = Properties(dictionary)
        if key not in self.arrays:
            self.arrays[key] = []
        self.arrays[key].append(props)
        _syn.add_to_array(self.obj, key, props.obj)


class Environment(CObject):
    def __init__(self, env):
        global _syn

        if type(env) is str:
            self.obj = _syn.load_env(env)
        elif type(env) is dict:
            env = Properties(env)
            self.obj = _syn.create_environment(env.obj)
            del env

    def save(self, filename):
        global _syn
        return _syn.save_env(self.obj, filename)


class Network(CObject):
    def __init__(self, net):
        global _syn

        if type(net) is str:
            self.obj = _syn.load_net(net)
        elif type(net) is dict:
            net = Properties(net)
            self.obj = _syn.create_network(net.obj)
            del net
        self.state = None

    def save(self, filename):
        global _syn
        return _syn.save_net(self.obj, filename)

    def run(self, environment, args):
        global _syn

        clear_env = clear_args = False

        if type(environment) is dict:
            environment = Environment(environment)
        if type(args) is dict:
            args = Properties(args)

        self.state = _syn.run(self.obj, environment.obj, self.state, args.obj);

        if self.state is None: print("Failed to run network!")
