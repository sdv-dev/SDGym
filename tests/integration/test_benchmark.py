import json

import sdgym


def test_identity():
    output = sdgym.run(
        synthesizers=['IdentitySynthesizer', 'IndependentSynthesizer', 'UniformSynthesizer'],
        datasets=['trains_v1', 'KRK_v1'],
    )

    assert not output.empty
    assert set(output['modality'].unique()) == {'single-table', 'multi-table'}

    scores = output.groupby('synthesizer').score.mean().sort_values()

    assert [
        'UniformSynthesizer',
        'IndependentSynthesizer',
        'IdentitySynthesizer',
    ] == scores.index.tolist()


def test_identity_jobs():
    jobs = [
        ('IdentitySynthesizer', 'trains_v1', 0),
        ('IndependentSynthesizer', 'trains_v1', 1),
        ('UniformSynthesizer', 'KRK_v1', 1),
    ]
    output = sdgym.run(jobs=jobs)

    assert not output.empty
    assert set(output['modality'].unique()) == {'single-table', 'multi-table'}

    columns = ['synthesizer', 'dataset', 'iteration']
    combinations = set(
        tuple(record)
        for record in output[columns].drop_duplicates().to_records(index=False)
    )

    assert combinations == set(jobs)


def test_json_synthesizer():
    synthesizer = {
        'name': 'synthesizer_name',
        'synthesizer': 'sdgym.synthesizers.ydata.PreprocessedVanillaGANSynthesizer',
        'modalities': ['single-table'],
        'init_kwargs': {'categorical_transformer': 'label_encoding'},
        'fit_kwargs': {'data': '$real_data'}
    }

    output = sdgym.run(
        synthesizers=[json.dumps(synthesizer)],
        datasets=['KRK_v1'],
        iterations=1,
    )

    assert set(output['synthesizer']) == {'synthesizer_name'}


def test_json_synthesizer_multi_table():
    synthesizer = {
        'name': 'HMA1',
        'synthesizer': 'sdv.relational.HMA1',
        'modalities': [
            'multi-table'
        ],
        'init_kwargs': {
            'metadata': '$metadata'
        },
        'fit_kwargs': {
            'tables': '$real_data'
        }
    }

    output = sdgym.run(
        synthesizers=[json.dumps(synthesizer)],
        datasets=['university_v1', 'trains_v1'],
        iterations=1,
    )

    # CSTest for `university_v1` is not valid because there are no categorical columns.
    valid_out = output.loc[~((output.dataset == 'university_v1') & (output.metric == 'CSTest'))]

    assert not valid_out.error.any()
