
def build_generator(source, target, model='drn'):
    if source == 'usps' or target == 'usps':
        from usps import Feature
        return Feature()

    if source == 'svhn' or target == 'svhn':
        from svhn2mnist import Feature
        return Feature()

    if source == 'synth':
        from syn2gtrsb import Feature
        return Feature()

    if source == 'appa' or target == 'real':
        from appa_real import Feature
        return Feature()

    if source == 'gta' or source == 'synthia':
        n_class = 20 if source == 'gta' else 16

        if model == 'fcnvgg':
            from vgg_fcn import FCN8sBase
            return FCN8sBase(n_class)

        if model == 'drn':
            from dilated_fcn import DRNSegBase
            return DRNSegBase(model_name='drn_d_105', n_class=n_class)


def build_classifier(source, target, model='drn'):
    if source == 'usps' or target == 'usps':
        from usps import Predictor
        return Predictor()

    if source == 'svhn' or target == 'svhn':
        from svhn2mnist import Predictor
        return Predictor()

    if source == 'synth':
        from syn2gtrsb import Predictor
        return Predictor()

    if source == 'appa' or target == 'real':
        from appa_real import Predictor
        return Predictor()

    if source == 'gta' or source == 'synthia':
        n_class = 20 if source == 'gta' else 16

        if model == 'fcnvgg':
            from vgg_fcn import FCN8sClassifier
            return FCN8sClassifier(n_class)

        if model == 'drn':
            from dilated_fcn import DRNSegPixelClassifier
            return DRNSegPixelClassifier(n_class=n_class)
