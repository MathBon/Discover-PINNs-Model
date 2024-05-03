

class Genome:
    def __init__(self):
        self.structure = {}
        self.structure['layer_num'] = None
        self.structure['residual_layer'] = None
        self.structure['neuron_num'] = None
        self.activation = None
        self.fitness = None
        self.error = None 
        self.generation = None
        self.id = None
        self.ori_generation = None  # original generation
        self.ori_id = None  # original id
        self.para_num = None

    def compare(self, other_genome): # judge whether the both genomes are the same

        same = False
        if self.structure['layer_num'] == other_genome.structure['layer_num'] and\
            self.structure['residual_layer'] == other_genome.structure['residual_layer'] and\
            self.structure['neuron_num'] == other_genome.structure['neuron_num'] and\
            self.activation.print(show_param=True) == other_genome.activation.print(show_param=True):

            same = True

        return same

    def __str__(self):
        # for print()
        att = 'Attributes:'+ '\n'
        att += 'generation: ' + (str(self.generation) if self.generation != None else '') + '\n'
        att += 'id: ' + ("{:d}".format(self.id) if self.id != None else '') + '\n'
        att += 'original generation: ' + ("{:d}".format(self.ori_generation) if self.ori_generation != None else '') + '\n'
        att += 'original id: ' + ("{:d}".format(self.ori_id) if self.ori_id != None else '') + '\n'
        att += 'fitness: ' + ("{:.3e}".format(self.fitness) if self.fitness != None else '') + '\n'
        att += 'error: ' + ("{:.3e}".format(self.error) if self.error != None else '') + '\n'
        att += 'para_num: ' + ("{:d}".format(self.para_num) if self.para_num != None else '') + '\n'
        att += 'layer_num: ' + ("{:d}".format(self.structure['layer_num']) if self.structure['layer_num'] != None else '') + '\n'
        att += 'residual_layer: ' + (str(self.structure['residual_layer']) if self.structure['residual_layer'] != None else '') + '\n'
        att += 'neuron_num: ' + ("{:d}".format(self.structure['neuron_num']) if self.structure['neuron_num'] != None else '') + '\n'
        att += 'activation: ' + (self.activation.print(show_param=True) if self.activation != None else '')

        return att
