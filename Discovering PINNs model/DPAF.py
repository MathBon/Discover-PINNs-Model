
# The code below is modified based on the paper titled 'Discovering Parametric Activation Functions'.
# We extend our gratitude to Garrett Bingham from the University of Texas at Austin for sharing the original code.

import copy
import numpy as np
import random
import torch
from config import config


class Function(object):
    def __init__(self, f, children, name='f', parametric=True):
        self.f = f
        self.parent = None
        self.children = children
        for c in self.children:
            c.parent = self
        self.name = name
        self.param_edges = []
        if parametric:
            self.parameterize(parametric)

    def parameterize(self, parametric):
        if parametric == True: # for random edges (parametric = True)
            # max one parameter per normal node; max two parameters per leaf node
            max_param_num = sum([2 if len(node.children) == 0 else 1 for node in self.list_nodes()])  # Maximum number of parameters that can be inserted
            num_params = random.randint(0, min(max_param_num, config.ACT_PARA_THOLD)) # random selection
            self.param_edges = random.sample(range(max_param_num), num_params)  # parameters are assigned to edges at random (e.g. [5, 0])
        else: # for the given parametric edges (e.g. parametric = [0, 2])
            assert type(parametric) == list
            self.param_edges = parametric

        for node in self.list_nodes():
            node.param_edges = self.param_edges  # for recursively printing the parameter edges when show_param is True, the same param_edges is reserved for each node

    def call(self, arg, params=None, param_idx=0):
        if params is None:
            if len(self.children) == 0:
                return self.f(arg)
            functions = [c.call(arg) for c in self.children]
            return self.f(*functions)
        else:
            if len(self.children) == 0:
                return params[param_idx] * self.f(params[param_idx+1] * arg)  # for leaf node
            elif len(self.children) == 1:
                c = self.children[0]
                functions = [c.call(arg, params, param_idx+1)]
                return params[param_idx] * self.f(*functions)
            else:
                assert len(self.children) == 2
                c0 = self.children[0]
                c1 = self.children[1]
                idx_offset = sum([2 if len(node.children) == 0 else 1 for node in c0.list_nodes()])  # Maximum number of parameters that can be accommodated by this node and the following
                functions = [c0.call(arg, params, param_idx+1),
                             c1.call(arg, params, param_idx+1+idx_offset)]
                return params[param_idx] * self.f(*functions)

    def depth(self):
        if len(self.children) == 0:
            return 1
        return 1+max(c.depth() for c in self.children)

    def print(self, show_param=True, param_idx=0):
        if show_param is False:
            if len(self.children) == 0:
                return self.name.replace('<x>', 'x')
            elif len(self.children) == 1:
                return self.name.replace('<x>', self.children[0].print(False, None))
            else:
                assert len(self.children) == 2
                return self.name.replace('<x>', self.children[0].print(False, None)).replace('<y>', self.children[1].print(False, None))
        elif show_param is True:
            prefix = 'L' # layer-wise
            if len(self.children) == 0:
                part1 = f'{prefix}{param_idx} * ' if param_idx in self.param_edges else ''
                part2 = f"{self.name.replace('<x>', f'{prefix}{param_idx+1} * x')}" if param_idx+1 in self.param_edges else f"{self.name.replace('<x>', 'x')}"
                return part1 + part2
            elif len(self.children) == 1:
                if param_idx in self.param_edges:
                    return f"{prefix}{param_idx} * {self.name.replace('<x>', f'{self.children[0].print(True, param_idx+1)}')}"
                else:
                    return f"{self.name.replace('<x>', f'{self.children[0].print(True, param_idx+1)}')}"
            else:
                assert len(self.children) == 2
                c = self.children[0]
                idx_offset = sum([2 if len(node.children) == 0 else 1 for node in c.list_nodes()])
                if param_idx in self.param_edges:
                    return f"{prefix}{param_idx} * {self.name.replace('<x>', self.children[0].print(True, param_idx+1)).replace('<y>', self.children[1].print(True, param_idx+1+idx_offset))}"
                else:
                    return f"{self.name.replace('<x>', self.children[0].print(True, param_idx+1)).replace('<y>', self.children[1].print(True, param_idx+1+idx_offset))}"

    def list_nodes(self):
        if len(self.children) == 0:
            return [self]
        return [self] + [node for c in self.children for node in c.list_nodes()]


    def mutate(self, mutation = None):

        max_param_num = sum([2 if len(node.children) == 0 else 1 for node in self.list_nodes()])
        max_permissible_num = min(max_param_num, config.ACT_PARA_THOLD)

        node_mutation = ['change', 'change_all', 'insert', 'remove']
        para_mutation = ['change_para', 'add_para', 'remove_para']

        mutation_type = random.choices(['node_mutation', 'para_mutation'], [len(node_mutation), len(para_mutation)])[0]

        if mutation is None:
            if mutation_type == 'node_mutation':
                mutation = random.choice(node_mutation)
            else:
                assert mutation_type == 'para_mutation'
                mutation = random.choice(para_mutation)

        if mutation == 'insert' and len(self.list_nodes()) >= config.ACT_MAX_NODES: # Too many existing nodes (the number of nodes may be larger than config.ACT_MAX_NODES after mutation)
            if len(self.list_nodes()) == 1:
                mutation = 'change'
            else:
                if len(self.list_nodes()) > config.ACT_MAX_NODES:
                    mutation = 'remove'
                else:
                    assert len(self.list_nodes()) == config.ACT_MAX_NODES
                    mutation = random.choice(['change', 'change_all', 'remove'])

        elif mutation == 'remove' and len(self.list_nodes()) == 1: # Only one node exists
            if config.ACT_MAX_NODES == 1:
                mutation = 'change'
            else:
                assert config.ACT_MAX_NODES > 1
                mutation = random.choice(['change', 'insert'])

        elif mutation in para_mutation:
            if max_permissible_num == 0:
                assert len(self.param_edges) == 0
                print('Mutation of activation function is cancelled')
                return
            else:
                if len(self.param_edges) == 0:
                    mutation = 'add_para'
                elif len(self.param_edges) == max_param_num:
                    mutation = 'remove_para'
                elif len(self.param_edges) == config.ACT_PARA_THOLD:
                    assert config.ACT_PARA_THOLD < max_param_num
                    mutation = random.choice(['change_para', 'remove_para'])

        print(f'----------Activation function before mutation: {self.print(show_param=True)}')
        
        if mutation == 'change':
            # randomly change one operator at one node
            unary_names = [u[0] for u in UNARY]
            binary_names = [b[0] for b in BINARY]
            random_node = random.choice(self.list_nodes())
            if random_node.name in unary_names:
                print('Changing one unary operator...')
                while True:
                    temp_name, temp_f = random.choice(UNARY)
                    if random_node.name != temp_name: # avoid invalid mutation
                        random_node.name, random_node.f = temp_name, temp_f
                        break
            elif random_node.name in binary_names:
                print('Changing one binary operator...')
                while True:
                    temp_name, temp_f = random.choice(BINARY)
                    if random_node.name != temp_name:
                        random_node.name, random_node.f = temp_name, temp_f
                        break
            else:
                print(f'Error: {random_node.name} is not a unary or binary operator in the search space.')

        elif mutation == 'change_all':
            # randomly change all operators
            print('Changing all operators...')
            unary_names = [u[0] for u in UNARY]
            binary_names = [b[0] for b in BINARY]
            for node in self.list_nodes():
                if node.name in unary_names:
                    node.name, node.f = random.choice(UNARY)
                elif node.name in binary_names:
                    node.name, node.f = random.choice(BINARY)
                else:
                    print(f'Error: {node.name} is not a unary or binary operator in the search space.')

        elif mutation == 'insert':
            # the new edge, which connects to the lower node of the original edge, inherits the parametric information from the original edge;
            # the other new edges are not assigned trainable parameters
            leaves_num = sum(len(node.children) == 0 for node in self.list_nodes())
            insert_range = random.choices(['leaf_up', 'leaf_down'], [len(self.list_nodes()), leaves_num])[0]
            operator_type = random.choices(['unary', 'binary'], [len(UNARY), len(BINARY)])[0]  # each operator is chosen with equal probability

            if insert_range == 'leaf_down':  # insert a new node between the input and a leaf node
                # new leaf
                leaves_idx = [node_idx for node_idx, node in enumerate(self.list_nodes()) if len(node.children) == 0]
                leaf_n = random.randint(0, len(leaves_idx)-1)
                leaf_parent = self.list_nodes()[leaves_idx[leaf_n]]
                edge_idx = leaves_idx[leaf_n] + leaf_n + 1

                print(f'Inserting a new {operator_type} leaf...')
                if operator_type == 'unary':
                    name, f = random.choice(UNARY)
                    new_node = Function(f, [], name)
                    new_node.parent = leaf_parent
                    leaf_parent.children = [new_node]
                    self.param_edges = [i if i < edge_idx else i + 1  for i in self.param_edges]
                elif operator_type == 'binary': # Different from the paper 'DPAF', which keeps the computing result unchanged before and after mutation
                    identity_F = Function(identity, [], 'linear(<x>)')
                    name_u, f_u = random.choice(UNARY)
                    rand_F = Function(f_u, [], name_u)
                    name_b, f_b = random.choice(BINARY)
                    new_param_edges = []
                    if random.random() < 0.5:
                        new_node = Function(f_b, [identity_F, rand_F], name_b)
                        for i in self.param_edges:
                            if i < edge_idx:
                                new_param_edges.append(i)
                            elif i == edge_idx:
                                new_param_edges.append(i + 2)
                            elif i > edge_idx:
                                new_param_edges.append(i + 4)
                    else:
                        new_node = Function(f_b, [rand_F, identity_F], name_b)
                        for i in self.param_edges:
                            if i < edge_idx:
                                new_param_edges.append(i)
                            elif i >= edge_idx:
                                new_param_edges.append(i + 4)

                    new_node.parent = leaf_parent
                    leaf_parent.children = [new_node]
                    self.param_edges = new_param_edges

            else:
                assert insert_range == 'leaf_up' # insert a new root or intermediate node
                random_node = random.choice(self.list_nodes())
                node_idx = [node_idx for node_idx, node in enumerate(self.list_nodes()) if node is random_node][0]
                leaves_idx = [node_idx for node_idx, node in enumerate(self.list_nodes()) if len(node.children) == 0]
                previous_leaves_num = sum(i < node_idx for i in leaves_idx)
                edge_idx = node_idx + previous_leaves_num

                if node_idx == 0:
                    # new root
                    print(f'Inserting a new {operator_type} root...')
                    if operator_type == 'unary':
                        self.children = [copy.deepcopy(self)]
                        self.children[0].parent = self
                        self.name, self.f = random.choice(UNARY)
                        self.param_edges = [i + 1 for i in self.param_edges]
                    elif operator_type == 'binary':  # Different from the paper 'DPAF', which keeps the computing result unchanged before and after mutation
                        name_u, f_u = random.choice(UNARY)
                        rand_F = Function(f_u, [], name_u)
                        if random.random() < 0.5:
                            self.children = [copy.deepcopy(self), rand_F]
                            self.param_edges = [i + 1 for i in self.param_edges]
                        else:
                            self.children = [rand_F, copy.deepcopy(self)]
                            self.param_edges = [i + 3 for i in self.param_edges]
                        self.children[0].parent = self
                        self.children[1].parent = self
                        name_b, f_b = random.choice(BINARY)
                        self.name, self.f = name_b, f_b
                else:
                    # new intermediate node
                    print(f'Inserting a new {operator_type} intermediate node...')

                    if operator_type == 'unary':
                        name, f = random.choice(UNARY)
                        new_node = Function(f, [], name)
                        child_idx = [i for i in range(len(random_node.parent.children)) if random_node.parent.children[i] is random_node][0] # the ranking of the random_node in its parents' children
                        random_node.parent.children[child_idx] = new_node
                        new_node.parent = random_node.parent
                        random_node.parent = new_node
                        new_node.children = [random_node]
                        self.param_edges = [i if i < edge_idx else i + 1 for i in self.param_edges]

                    elif operator_type == 'binary':  # Different from the paper 'DPAF', which keeps the computing result unchanged before and after mutation
                        name_b, f_b = random.choice(BINARY)
                        new_node_b = Function(f_b, [], name_b)
                        name_u, f_u = random.choice(UNARY)
                        new_node_u = Function(f_u, [], name_u)
                        child_idx = [i for i in range(len(random_node.parent.children)) if random_node.parent.children[i] is random_node][0]
                        random_node.parent.children[child_idx] = new_node_b
                        new_node_b.parent = random_node.parent
                        random_node.parent = new_node_b
                        new_node_u.parent = new_node_b
                        if random.random() < 0.5:
                            new_node_b.children = [random_node, new_node_u]
                            left_edges_num = sum([2 if len(node.children) == 0 else 1 for node in random_node.list_nodes()])
                            new_param_edges = []
                            for i in self.param_edges:
                                if i < edge_idx:
                                    new_param_edges.append(i)
                                elif edge_idx <= i and i < edge_idx + left_edges_num:
                                    new_param_edges.append(i + 1)
                                elif i >= edge_idx + left_edges_num:
                                    new_param_edges.append(i + 3)
                            self.param_edges = new_param_edges
                        else:
                            new_node_b.children = [new_node_u, random_node]
                            self.param_edges = [i if i < edge_idx else i + 3 for i in self.param_edges]
        elif mutation == 'remove': # choose a random node, remove it
            random_node = random.choice(self.list_nodes())
            node_idx = [node_idx for node_idx, node in enumerate(self.list_nodes()) if node is random_node][0]
            leaves_idx = [node_idx for node_idx, node in enumerate(self.list_nodes()) if len(node.children) == 0]
            previous_leaves_num = sum(i < node_idx for i in leaves_idx)
            edge_idx = node_idx + previous_leaves_num

            if random_node.parent is None:
                assert random_node == self
                if len(self.children) == 1:
                    print('Removing unary root...')
                    self.name = self.children[0].name
                    self.f = self.children[0].f
                    for grandchild in self.children[0].children: # may be empty
                        grandchild.parent = self
                    self.children = self.children[0].children # may be empty
                    self.param_edges = [i - 1 for i in self.param_edges if i > 0]
                else:
                    assert len(self.children) == 2
                    child_to_keep = random.choice([0, 1])
                    print(f'Removing binary root; keeping {"right" if child_to_keep else "left"} child...')
                    left_edges_num = sum([2 if len(node.children) == 0 else 1 for node in self.children[0].list_nodes()]) # number of edges of left side
                    self.name = self.children[child_to_keep].name
                    self.f = self.children[child_to_keep].f
                    for grandchild in self.children[child_to_keep].children:  # may be empty
                        grandchild.parent = self
                    self.children = self.children[child_to_keep].children  # may be empty

                    if child_to_keep == 0:
                        self.param_edges = [i - 1 for i in self.param_edges if i > 0 and i <= left_edges_num]
                    else:
                        assert child_to_keep == 1
                        self.param_edges = [i - left_edges_num - 1 for i in self.param_edges if i > left_edges_num]

            elif len(random_node.children) == 0:
                print('Removing leaf...')
                if len(random_node.parent.children) == 2:
                    # parent is binary, replace leaf with identity
                    random_node.name = 'linear(<x>)'
                    random_node.f = identity
                    if edge_idx in self.param_edges:
                        self.param_edges.remove(edge_idx)
                else:
                    assert len(random_node.parent.children) == 1
                    new_param_edges = []
                    for i in self.param_edges:
                        if i < edge_idx:
                            new_param_edges.append(i)
                        elif i > edge_idx:
                            new_param_edges.append(i-1)
                    self.param_edges = new_param_edges
                    # parent is unary, delete leaf
                    random_node.parent.children = []

            elif len(random_node.children) == 1:
                print('Removing unary intermediate node...')
                new_param_edges = []
                for i in self.param_edges:
                    if i < edge_idx:
                        new_param_edges.append(i)
                    elif i > edge_idx:
                        new_param_edges.append(i - 1)
                self.param_edges = new_param_edges

                random_node.children[0].parent = random_node.parent
                child_idx = [i for i in range(len(random_node.parent.children)) if random_node.parent.children[i] is random_node][0]
                random_node.parent.children[child_idx] = random_node.children[0]

            else:
                assert len(random_node.children) == 2
                child_to_keep = random.choice([0, 1])
                print(f'Removing binary intermediate node; keeping {"right" if child_to_keep else "left"} child...')
                left_edges_num = sum([2 if len(node.children) == 0 else 1 for node in random_node.children[0].list_nodes()])
                right_edges_num = sum([2 if len(node.children) == 0 else 1 for node in random_node.children[1].list_nodes()])
                new_param_edges = []
                if child_to_keep == 0:
                    for i in self.param_edges:
                        if i < edge_idx:
                            new_param_edges.append(i)
                        elif i > edge_idx and i <= edge_idx + left_edges_num:
                            new_param_edges.append(i - 1)
                        elif i > edge_idx + left_edges_num + right_edges_num:
                            new_param_edges.append(i - right_edges_num - 1)
                else:
                    assert child_to_keep == 1
                    for i in self.param_edges:
                        if i < edge_idx:
                            new_param_edges.append(i)
                        elif i > edge_idx + left_edges_num:
                            new_param_edges.append(i - left_edges_num -1)
                self.param_edges = new_param_edges

                random_node.children[child_to_keep].parent = random_node.parent
                child_idx = [i for i in range(len(random_node.parent.children)) if random_node.parent.children[i] is random_node][0]
                random_node.parent.children[child_idx] = random_node.children[child_to_keep]

        elif mutation == 'change_para':  # the number of parametric edges is unchanged
            assert len(self.param_edges) < max_param_num
            assert len(self.param_edges) > 0
            new_edge = random.choice([i for i in list(range(max_param_num)) if i not in self.param_edges])
            remove_idx = random.choice(range(len(self.param_edges)))
            self.param_edges.pop(remove_idx)
            self.param_edges.append(new_edge)
            print('Changing one parametric edge...')

        elif mutation == 'add_para':
            assert len(self.param_edges) < max_permissible_num
            new_edge = random.choice([i for i in list(range(max_param_num)) if i not in self.param_edges])
            self.param_edges.append(new_edge)
            print('Adding one parametric edge...')

        else:
            assert mutation == 'remove_para'
            assert len(self.param_edges) > 0
            remove_idx = random.choice(range(len(self.param_edges)))
            self.param_edges.pop(remove_idx)
            print('Removing one parametric edge...')


        for node in self.list_nodes():
            node.param_edges = self.param_edges

        # test code
        assert len(set(self.param_edges)) == len(self.param_edges)
        new_max_param_num = sum([2 if len(node.children) == 0 else 1 for node in self.list_nodes()])  # max_param_num may be changed after mutation
        assert sum(i > new_max_param_num-1 for i in self.param_edges) == 0
        assert sum(i < 0 for i in self.param_edges) == 0
        assert len(self.param_edges) <= new_max_param_num

        print(f'++++++++++Activation function after mutation: {self.print(show_param=True)}')



def identity(x):
    return x
def recu(x):
    return torch.relu(x)**3
def fixedswish(x):
    return x * torch.sigmoid(x)
def exp_p_1(x):
    return 1.0 + torch.exp(x)
def expn_p_1(x):
    return 1.0 + torch.exp(-x)
def exp_s_1(x):
    return torch.exp(x) - 1.0
def exp_p_expn(x):
    return torch.exp(x) + torch.exp(-x)
def exp_s_expn(x):
    return torch.exp(x) - torch.exp(-x)


UNARY = [
         ('exp_p_1(<x>)'      , exp_p_1),
         ('expn_p_1(<x>)'     , expn_p_1),
         ('exp_s_1(<x>)'      , exp_s_1),
         ('exp_p_expn(<x>)'   , exp_p_expn),
         ('exp_s_expn(<x>)'   , exp_s_expn),
         ('linear(<x>)'       , identity),
         ('sin(<x>)'          , torch.sin),
         ('sinh(<x>)'         , torch.sinh),
         ('asinh(<x>)'        , torch.asinh),
         ('cos(<x>)'          , torch.cos),
         ('cosh(<x>)'         , torch.cosh),
         ('tanh(<x>)'         , torch.tanh),
         ('atan(<x>)'         , torch.atan),
         ('erf(<x>)'          , torch.erf),
         ('erfc(<x>)'         , torch.erfc),
         ('exp(<x>)'          , torch.exp),
         ('-(<x>)'            , torch.negative),
         ('(<x>)^-1'          , torch.reciprocal),
         ('sigmoid(<x>)'      , torch.sigmoid),
         ('softsign(<x>)'     , torch.nn.Softsign()),
         ('fixedswish(<x>)'   , fixedswish),
         ('(<x>)^2'           , torch.square),
         ('softplus(<x>)'     , torch.nn.Softplus()),
        ]

BINARY = [
          ('((<x>) + (<y>))'   , torch.add),
          ('((<x>) - (<y>))'   , torch.subtract),
          ('((<x>) * (<y>))'   , torch.multiply),
          ('((<x>) / (<y>))'   , torch.divide),
          ('max(<x>, <y>)'     , torch.maximum),
          ('min(<x>, <y>)'     , torch.minimum),
         ]


def init_actfun(actfun=None, mutations_num=0):

    if random.random() < config.INITIAL_BASEACT_RATIO :

        actname = random.choices(['tanh', 'atan', 'sin', 'cos', 'asinh', 'sigmoid', 'recu', 'fixedswish',
                                  'adapttanh', 'adaptsin', 'adaptcos', 'adaptsigmoid', 'adaptswish'])[0]

        if actname == 'tanh':
            if random.random() < 0.5:
                Fn = Function(torch.tanh, [], 'tanh(<x>)', parametric=None)
            else:
                F1 = Function(exp_s_expn, [], 'exp_s_expn(<x>)')
                F2 = Function(exp_p_expn, [], 'exp_p_expn(<x>)')
                Fn = Function(torch.divide, [F1, F2], '((<x>) / (<y>))', parametric=None)

        elif actname == 'atan':
            Fn = Function(torch.atan, [], 'atan(<x>)', parametric=None)

        elif actname == 'sin':
            Fn = Function(torch.sin, [], 'sin(<x>)', parametric=None)

        elif actname == 'cos':
            Fn = Function(torch.cos, [], 'cos(<x>)', parametric=None)

        elif actname == 'asinh':
            Fn = Function(torch.asinh, [], 'asinh(<x>)', parametric=None)

        elif actname == 'sigmoid':
            if random.random() < 0.5:
                Fn = Function(torch.sigmoid, [], 'sigmoid(<x>)', parametric=None)
            else:
                F1 = Function(expn_p_1, [], 'expn_p_1(<x>)')
                Fn = Function(torch.reciprocal, [F1], '(<x>)^-1', parametric=None)

        elif actname == 'recu':
            Fn = Function(recu, [], 'recu(<x>)', parametric=None)

        elif actname == 'fixedswish':
            if random.random() < 0.5:
                Fn = Function(fixedswish, [], 'fixedswish(<x>)', parametric=None)
            else:
                F1 = Function(identity, [], 'linear(<x>)')
                F2 = Function(expn_p_1, [], 'expn_p_1(<x>)')
                Fn = Function(torch.divide, [F1, F2], '((<x>) / (<y>))', parametric=None)

        elif actname == 'adapttanh':
            Fn = Function(torch.tanh, [], 'tanh(<x>)', parametric=[1])

        elif actname == 'adaptsin':
            Fn = Function(torch.sin, [], 'sin(<x>)', parametric=[1])

        elif actname == 'adaptcos':
            Fn = Function(torch.cos, [], 'cos(<x>)', parametric=[1])

        elif actname == 'adaptsigmoid':
            if random.random() < 0.5:
                Fn = Function(torch.sigmoid, [], 'sigmoid(<x>)', parametric=[1])
            else:
                F1 = Function(expn_p_1, [], 'expn_p_1(<x>)')
                Fn = Function(torch.reciprocal, [F1], '(<x>)^-1', parametric=[2])

        else:
            assert actname == 'adaptswish'
            F1 = Function(identity, [], 'linear(<x>)')
            F2 = Function(torch.sigmoid, [], 'sigmoid(<x>)')
            Fn = Function(torch.multiply, [F1, F2], '((<x>) * (<y>))', parametric=[4])


    else:
        if random.random() < 0.5:
            # unary2(unary1(x))
            name_u1, f_u1 = random.choice(UNARY)
            F1 = Function(f_u1, [], name_u1)
            name_u2, f_u2 = random.choice(UNARY)
            Fn = Function(f_u2, [F1], name_u2, parametric=True)
        else:
            # binary(unary1(x), unary2(x))
            name_u1, f_u1 = random.choice(UNARY)
            F1 = Function(f_u1, [], name_u1)
            name_u2, f_u2 = random.choice(UNARY)
            F2 = Function(f_u2, [], name_u2)
            name_b, f_b = random.choice(BINARY)
            Fn = Function(f_b, [F1, F2], name_b, parametric=True)


    for _ in range(mutations_num):
        Fn.mutate()

    return Fn


if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
