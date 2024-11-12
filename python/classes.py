from itertools import chain


type_models_dict = {
    'airplane': (
        'A10', 'A400M', 'AG600', 'An124', 'An22', 'An225', 'An72', 'AV8B',
        'B1', 'B2', 'B21', 'B52', 'Be200', 'C130', 'C17', 'C2', 'C390', 'C5',
        'CL415', 'E2', 'E7', 'EF2000', 'F117', 'F14', 'F15', 'F16', 'F18',
        'F22', 'F35', 'F4', 'H6', 'J10', 'J20', 'JAS39', 'JF17', 'JH7',
        'KC135', 'KF21', 'KJ600', 'Mig29', 'Mig31', 'Mirage2000', 'MQ9',
        'P3', 'Rafale', 'RQ4', 'SR71', 'Su24', 'Su25', 'Su34', 'Su57',
        'TB001', 'TB2', 'Tornado', 'Tu160', 'Tu22M', 'Tu95', 'U2', 'US2',
        'Vulcan', 'WZ7', 'XB70', 'Y20', 'YF23'
    ),
    'helicopter': (
        'AH64', 'CH47', 'Ka27', 'Ka52', 'Mi24', 'Mi26', 'Mi28', 'UH60',
        'V22', 'Z19'
    )
}


def model_type_dict_gen(airplane_label, helicopter_label):
    """
    Generate a dictionary where keys are model names and values are type labels.

    Parameters:
        airplane_label: Label for airplanes
        helicopter_label: Label for helicopters

    Returns:
        dict: A dictionary with class names as keys and type labels as values
    """
    result = {model_name: airplane_label for model_name in type_models_dict['airplane']}
    result.update({model_name: helicopter_label for model_name in type_models_dict['helicopter']})
    return result


model_names = tuple(chain.from_iterable(type_models_dict.values()))
model_type_index_dict = model_type_dict_gen(1, 0)
model_type_name_dict = model_type_dict_gen('airplane', 'helicopter')


class_names = {
    'models': model_names,
    'types': ('airplane', 'helicopter')
}

classes_map = {
    'models': {model_name: model_name for model_name in model_names},
    'types': model_type_name_dict
}
