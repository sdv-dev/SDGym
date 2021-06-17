import json

import sdgym


def test_identity():
    output = sdgym.run(
        synthesizers=['Identity', 'Independent', 'Uniform'],
        datasets=['trains_v1', 'KRK_v1'],
    )

    assert not output.empty
    assert set(output['modality'].unique()) == {'single-table', 'multi-table'}

    scores = output.groupby('synthesizer').score.mean().sort_values()

    assert ['Uniform', 'Independent', 'Identity'] == scores.index.tolist()


def test_identity_jobs():
    jobs = [
        ('Identity', 'trains_v1', 0),
        ('Independent', 'trains_v1', 1),
        ('Uniform', 'KRK_v1', 1),
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
        "name": "synthesizer_name",
        "synthesizer": "sdgym.synthesizers.ydata.PreprocessedVanillaGAN",
        "modalities": ["single-table"],
        "init_kwargs": {"categorical_transformer": "label_encoding"},
        "fit_kwargs": {"data": "$real_data"}
    }

    output = sdgym.run(
        synthesizers=[json.dumps(synthesizer)],
        datasets=['KRK_v1'],
        iterations=1,
    )

    assert set(output['synthesizer']) == {"synthesizer_name"}
