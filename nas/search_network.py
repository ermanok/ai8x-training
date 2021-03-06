###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Application to run evolutionary search over trained Once For All model.
"""

import argparse
import os
import fnmatch
from pydoc import locate
import sys
import inspect

import torch
from torch.utils.data import DataLoader

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import ai8x  # pylint: disable=wrong-import-position
import parse_ofa_yaml # pylint: disable=wrong-import-position
import nas_utils # pylint: disable=wrong-import-position
from evo_search import EvolutionSearch # pylint: disable=wrong-import-position


def parse_args(model_names, dataset_names):
    """Return the parsed arguments"""
    parser = argparse.ArgumentParser(description='Evolutionary search for a trained once ' \
                                                 'for all model')
    parser.add_argument('--model_path', metavar='DIR', required=True, help='path to model ' \
                                                                           'checkpoint')
    parser.add_argument('--arch', '-a', '--model', metavar='ARCH', required=True,
                        type=lambda s: s.lower(), dest='arch', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names))
    parser.add_argument('--dataset', metavar='S', required=True, choices=dataset_names,
                        help='dataset: ' + ' | '.join(dataset_names))
    parser.add_argument('--data', metavar='DIR', default='data', help='path to dataset')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--no-bias', action='store_true', default=False,
                        help='for models that support both bias and no bias, set the '
                             '`use bias` flag to true')

    parser.add_argument('--ofa-policy', dest='ofa_policy', required=True,
                        help='path to YAML file that defines the OFA ' \
                             '(once for all training) policy')

    return parser.parse_args()


def get_evo_search_params(ofa_policy):
    """Get parameters used for evolutionary search from yaml file"""
    evo_search_params = {'population_size': 100, 'prob_mutation': 0.1, 'ratio_mutation': 0.5,
                         'ratio_parent': 0.25, 'num_iter': 500,
                         'constraints': {'max_num_weights': 4.5e5}}

    if 'evolution_search' in ofa_policy:
        for key in evo_search_params:
            if key in ofa_policy['evolution_search']:
                evo_search_params[key] = ofa_policy['evolution_search'][key]

    return evo_search_params

def load_models():
    """Dynamically load models"""
    supported_models = []
    model_names = []

    for _, _, files in sorted(os.walk('models')):
        for name in sorted(files):
            if fnmatch.fnmatch(name, '*.py'):
                fn = 'models.' + name[:-3]
                m = locate(fn)
                try:
                    for i in m.models:
                        i['module'] = fn
                    supported_models += m.models
                    model_names += [item['name'] for item in m.models]
                except AttributeError:
                    # Skip files that don't have 'models' or 'models.name'
                    pass

    return supported_models, model_names


def load_datasets():
    """Dynamically load datasets"""
    supported_sources = []
    dataset_names = []

    for _, _, files in sorted(os.walk('datasets')):
        for name in sorted(files):
            if fnmatch.fnmatch(name, '*.py'):
                ds = locate('datasets.' + name[:-3])
                try:
                    supported_sources += ds.datasets
                    dataset_names += [item['name'] for item in ds.datasets]
                except AttributeError:
                    # Skip files that don't have 'datasets' or 'datasets.name'
                    pass

    return supported_sources, dataset_names


def get_data_loaders(supported_sources, args):
    """Dynamically loads data loaders"""
    selected_source = next((item for item in supported_sources if item['name'] == args.dataset))
    labels = selected_source['output']
    num_classes = len(labels)
    if num_classes == 1 or ('regression' in selected_source and selected_source['regression']):
        args.regression = True
    else:
        args.regression = False

    args.dimensions = selected_source['input']
    args.num_classes = len(selected_source['output'])

    train_dataset, val_dataset = selected_source['loader']((args.data, args))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    return train_loader, val_loader


def create_model(supported_models, args):
    """Create the model"""
    module = next(item for item in supported_models if item['name'] == args.arch)

    Model = locate(module['module'] + '.' + args.arch)

    if not Model:
        raise RuntimeError("Model " + args.arch + " not found\n")

    if module['dim'] > 1 and module['min_input'] > args.dimensions[2]:
        model = Model(pretrained=False, num_classes=args.num_classes,
                      num_channels=args.dimensions[0],
                      dimensions=(args.dimensions[1], args.dimensions[2]),
                      padding=(module['min_input'] - args.dimensions[2] + 1) // 2,
                      bias=not args.no_bias).to(args.device)
    else:
        model = Model(pretrained=False, num_classes=args.num_classes,
                      num_channels=args.dimensions[0],
                      dimensions=(args.dimensions[1], args.dimensions[2]),
                      bias=not args.no_bias).to(args.device)

    return model


def main():
    """Main routine"""
    ai8x.set_device(device=85, simulate=False, round_avg=False, verbose=False)

    supported_models, model_names = load_models()
    supported_sources, dataset_names = load_datasets()

    args = parse_args(model_names, dataset_names)
    args.truncate_testset = False
    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if use_cuda else "cpu")
    args.act_mode_8bit = False

    # Get policy for once for all training policy
    ofa_policy = parse_ofa_yaml.parse(args.ofa_policy) \
                 if args.ofa_policy.lower() != '' else None

    # Get data loaders
    train_loader, val_loader = get_data_loaders(supported_sources, args)

    # Load model
    model = create_model(supported_models, args)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])

    # Calculate full model accuracy
    full_model_acc = nas_utils.calc_accuracy(None, model, train_loader, val_loader, args.device)
    print(f'Model Accuracy: {100*full_model_acc: .3f}%')

    # Run evolutionary search to find proper networks
    evo_search_params = get_evo_search_params(ofa_policy)
    evo_search = EvolutionSearch(population_size=evo_search_params['population_size'],
                                 prob_mutation=evo_search_params['prob_mutation'],
                                 ratio_mutation=evo_search_params['ratio_mutation'],
                                 ratio_parent=evo_search_params['ratio_parent'],
                                 num_iter=evo_search_params['num_iter'])
    evo_search.set_model(model)
    best_arch, best_acc = evo_search.run(evo_search_params['constraints'], train_loader,
                                         val_loader, args.device)
    print(best_arch, best_acc)

if __name__ == '__main__':
    main()
