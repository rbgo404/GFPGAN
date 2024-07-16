INPUT_SCHEMA = {
    "img_url": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Abraham_Lincoln_O-77_matte_collodion_print.jpg/1024px-Abraham_Lincoln_O-77_matte_collodion_print.jpg"]
    },
    "scale": {
        'datatype': 'INT8',
        'required': True,
        'shape': [1],
        'example': [2]
    }
}