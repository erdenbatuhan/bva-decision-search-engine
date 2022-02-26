import random


def split_data_balanced(inputs, type_key, val_test_size=.1):
    """
    Splits the data in a balanced way

    :param inputs: Inputs as a dictionary where IDs and items are keys and values respectively
    :param type_key: Key of the type attribute that the data will be balanced around (e.g. outcome)
    :param val_test_size: Percentage of the validation and test data size
    :return: Tuple containing the training, validation and test data
    """
    distinct_types = set([item[type_key] for _, item in inputs.items()])
    inputs_by_type = {distinct_type: [] for distinct_type in distinct_types}

    # Add inputs to the dictionary based on their types
    for key, item in inputs.items():
        inputs_by_type[item[type_key]].append(key)

    # Shuffle inputs for each type
    for _, item in inputs_by_type.items():
        random.shuffle(item)

    train, val, test = [], [], []
    size_per_type = int(len(inputs) / len(distinct_types))

    # Populate the data split
    for distinct_type in distinct_types:
        step = int(val_test_size * size_per_type)

        test = test + inputs_by_type[distinct_type][0:step]
        val = val + inputs_by_type[distinct_type][step:2 * step]
        train = train + inputs_by_type[distinct_type][2 * step:]

    return train, val, test


def assert_balanced_split(data_tuple, identifier_key, type_key):
    """
    Asserts if a split is balanced

    :param data_tuple: Tuple containing the training, validation and test data
    :param identifier_key: Key of the ID attribute (e.g. document)
    :param type_key: Key of the type attribute (Data is balanced using this key) (e.g. outcome)
    """

    distinct_types = set([item[type_key] for item in data_tuple[0]])

    for data in data_tuple:
        item_types = set([(item[identifier_key], item[type_key]) for item in data])

        type_counts = [
            sum([item_type == distinct_type for (_, item_type) in item_types])
            for distinct_type in distinct_types
        ]
        distinct_type_counts = list(set(type_counts))

        assert len(distinct_type_counts) == 1 or (
                len(distinct_type_counts) == 2 and
                abs(distinct_type_counts[1] - distinct_type_counts[0]) <= 1
        ), "The split is not balanced!"

