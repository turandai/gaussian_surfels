import csv
import os

forbidden_buildings = ['mosquito', 'tansboro']

# The 'small-image' dataset has some misssing buildings. Either we can ignore those buildings or we can regenerate the dataset.
forbidden_buildings = ['newfields', 'ludlowville', 'goodyear', 'castroville']  # missing RGB
# forbidden_buildings = ['mosquito', 'tansboro', 'tomkins', 'darnestown', 'brinnon']
# We do not have the rgb data for tomkins, darnestown, brinnon

forbidden_buildings += ['rough'] # Contains some wrong view-points

# SPLIT_TO_NUM_IMAGES = {
#     'supersmall': 14575,
#     'tiny': 262745,
#     'fullplus': 3349691,
# }

def get_splits(split_path):
    with open(split_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        train_list = []
        val_list = []
        test_list = []

        for row in readCSV:
            name, is_train, is_val, is_test = row
            if name in forbidden_buildings:
                continue
            if is_train == '1':
                train_list.append(name)
            if is_val == '1':
                val_list.append(name)
            if is_test == '1':
                test_list.append(name)
    return {
        'train': sorted(train_list),
        'val': sorted(val_list),
        'test': sorted(test_list)
    }


# Taskonomy
subsets = ['debug', 'tiny', 'medium', 'full', 'fullplus']
taskonomy_split_files = {s:  os.path.join(os.path.dirname(__file__),
                                'splits',
                                'train_val_test_{}.csv'.format(s.lower()))
               for s in subsets}

taskonomy_split_to_buildings = {s: get_splits(taskonomy_split_files[s]) for s in subsets}

taskonomy_flat_split_to_buildings = {}
for subset in taskonomy_split_to_buildings:
    for split, buildings in taskonomy_split_to_buildings[subset].items():
        taskonomy_flat_split_to_buildings[subset + '-' + split] = buildings


# Replica
replica_split_file = os.path.join(os.path.dirname(__file__), 'splits', 'train_val_test_replica.csv')
replica_flat_split_to_buildings = get_splits(replica_split_file)

# GSO
gso_split_file = os.path.join(os.path.dirname(__file__), 'splits', 'train_val_test_gso.csv')
gso_flat_split_to_buildings = get_splits(gso_split_file)

# hypersim
hypersim_split_file = os.path.join(os.path.dirname(__file__), 'splits', 'train_val_test_hypersim.csv')
hypersim_flat_split_to_buildings = get_splits(hypersim_split_file)

# blendedMVS
blendedMVS_split_file = os.path.join(os.path.dirname(__file__), 'splits', 'train_val_test_blendedMVS.csv')
blendedMVS_flat_split_to_buildings = get_splits(blendedMVS_split_file)

