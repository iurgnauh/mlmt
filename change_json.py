import json
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument('--input_json', type=str, help='The original json config file')
arg_parser.add_argument('--out_json', type=str, default='temp.json', help='The original json config file')
arg_parser.add_argument('--sample_size', type=int, help='The size of the sample in the new json file')
args = arg_parser.parse_args()

orig_json = json.load(open(args.input_json, 'r'))
print('The orig json sample size is {0}, will be changed to {1}'.format(orig_json['datasets'][0]['sample'], args.sample_size))
orig_json['datasets'][0]['sample'] = args.sample_size

json.dump(orig_json, open(args.out_json, 'w'))
print('The target json has been save to {0}'.format(args.out_json))
