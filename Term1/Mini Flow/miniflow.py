'''
!@brief Simple implementation of a neural network
'''



class Node (object):
    def__init__(self, inbound_nodes = []):
    '''
    @brief Nodes are building block of neural networks
    @params list of input nodes

    '''

    # Node(s) from which this Node receives values
    self.inbound_nodes = inbound_nodes
    # Node(s) to which this Node passes values
    self.outbound_nodes = []
    # For each inbound_node, add the current Node as an outbound_node.
    for n in self.inbound_nodes:
        n.outbound_nodes.append(self)
    # A calculated value
    self.value = None


class Input(Node):
    def__init__(self):
    '''
    @brief Input nodes do not calculate values 
    and have no parent (input) nodes
    
    '''
    
    # create instance with empty input list
    Node.__init__(self)

    '''
    @brief Set value fron an input
        
    Unlike the other subclasses of Node, the Input subclass does 
    not actually calculate anything. The Input subclass just 
    holds a value, such as a data feature or a model parameter 
    (weight/bias).    

    @param Value
    '''

    def forward(self, value=None):
        if value is not None:
            self.value = value


class Add(Node):
    def__init__(self,x ,y):
    '''
    @brief Node subclass, adds output from two nodes
    and have no parent (input) nodes

    @params node 1 x, node 2 y
    
    '''
    
    # create instance with two input nodes in input list
    Node.__init__(self, [x,y])

    '''
    @brief Adds values of all inbound nodes
    '''
    def forward(self):
        value = 0
        for n in self.inbound_nodes:
            value += n.value
        self.value = value


class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """
        inputs = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value
        self.value = bias
        for x, w in zip(inputs, weights):
            self.value += x * w


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value

