import numpy as np
import pandas as pd

from sdgym.synthesizers.uniform import UniformSynthesizer


class TestUniformSynthesizer:
    def test_uniform_synthesizer_sdtypes(self):
        """Ensure that sdtypes uniform are taken from metadata instead of inferred."""
        uniform_synthesizer = UniformSynthesizer()
        metadata = {
            'primary_key': 'guest_email',
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
            'columns': {
                'guest_email': {'sdtype': 'email', 'pii': True},
                'has_rewards': {'sdtype': 'boolean'},
                'room_type': {'sdtype': 'categorical'},
                'amenities_fee': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                'room_rate': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                'billing_address': {'sdtype': 'address', 'pii': True},
                'credit_card_number': {'sdtype': 'credit_card_number', 'pii': True},
            },
        }

        data = {
            'guest_email': {
                0: 'michaelsanders@shaw.net',
                1: 'randy49@brown.biz',
                2: 'webermelissa@neal.com',
                3: 'gsims@terry.com',
                4: 'misty33@smith.biz',
            },
            'has_rewards': {0: False, 1: False, 2: True, 3: False, 4: False},
            'room_type': {0: 'BASIC', 1: 'BASIC', 2: 'DELUXE', 3: 'BASIC', 4: 'BASIC'},
            'amenities_fee': {0: 37.89, 1: 24.37, 2: 0.0, 3: np.nan, 4: 16.45},
            'checkin_date': {
                0: '27 Dec 2020',
                1: '30 Dec 2020',
                2: '17 Sep 2020',
                3: '28 Dec 2020',
                4: '05 Apr 2020',
            },
            'checkout_date': {
                0: '29 Dec 2020',
                1: '02 Jan 2021',
                2: '18 Sep 2020',
                3: '31 Dec 2020',
                4: np.nan,
            },
            'room_rate': {0: 131.23, 1: 114.43, 2: 368.33, 3: 115.61, 4: 122.41},
            'billing_address': {
                0: '49380 Rivers Street\nSpencerville, AK 68265',
                1: '88394 Boyle Meadows\nConleyberg, TN 22063',
                2: '0323 Lisa Station Apt. 208\nPort Thomas, LA 82585',
                3: '77 Massachusetts Ave\nCambridge, MA 02139',
                4: '1234 Corporate Drive\nBoston, MA 02116',
            },
            'credit_card_number': {
                0: 4075084747483975747,
                1: 180072822063468,
                2: 38983476971380,
                3: 4969551998845740,
                4: 3558512986488983,
            },
        }

        real_data = pd.DataFrame(data)
        synthesizer = uniform_synthesizer.get_trained_synthesizer(real_data, metadata)
        hyper_transformer_config = synthesizer[0].get_config()
        config_sdtypes = hyper_transformer_config['sdtypes']
        unknown_sdtypes = ['email', 'credit_card_number', 'address']
        for column in metadata['columns']:
            metadata_sdtype = metadata['columns'][column]['sdtype']
            # Only data types that are known are overridden by metadata
            if metadata_sdtype not in unknown_sdtypes:
                assert metadata_sdtype == config_sdtypes[column]
            else:
                assert config_sdtypes[column] == 'pii'
