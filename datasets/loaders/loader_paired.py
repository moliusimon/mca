from builtins import object


class LoaderPaired(object):
    def __init__(self, data_loader_A, data_loader_B):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B

    def __iter__(self):
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None

        try:
            A, A_paths = next(self.data_loader_A_iter)

        except StopIteration:
            if A is None or A_paths is None:
                raise StopIteration

        try:
            B, B_paths = next(self.data_loader_B_iter)
            if B.size()[0] < self.data_loader_B.batch_size:
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        except StopIteration:
            self.data_loader_B_iter = iter(self.data_loader_B)
            B, B_paths = next(self.data_loader_B_iter)

        self.iter += 1
        return {
            'S': A, 'S_label': A_paths,
            'T': B, 'T_label': B_paths
        }
